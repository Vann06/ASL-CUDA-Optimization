"""Augmentation fotométrico de CNN2 (brillo + contraste) en CUDA Python con Numba.

Es la misma política que ``CNN2Augmentation`` (pasos 4 y 5), pero aquí el reparto del
trabajo en la GPU lo decidimos nosotros en vez de PyTorch:

    brillo:     y = clamp(x + delta[n], 0, 1)
    contraste:  z = clamp((y - media[n, c]) * factor[n] + media[n, c], 0, 1)

``media[n, c]`` es la media espacial del canal ``c`` de la imagen ``n`` *después* del
brillo, así que el algoritmo tiene dos etapas con una dependencia entre ellas:

Kernel 1, ``brightness_mean_kernel`` (reducción)
    * grid  = ``N*C`` bloques (uno por canal de cada imagen: 32*3 = 96 en un batch).
    * block = 256 hilos. Cada hilo recorre los 64*64 = 4096 píxeles del canal con paso
      256 (16 píxeles por hilo; accesos contiguos entre hilos vecinos → *coalesced*),
      aplica el brillo, escribe el píxel y acumula una suma parcial.
    * Las 256 sumas parciales se reducen en memoria compartida en log2(256) = 8 pasos
      sincronizados con ``cuda.syncthreads()``; el hilo 0 escribe la media.

Kernel 2, ``contrast_kernel`` (elemento a elemento)
    * block = 16x16 hilos (256), uno por píxel de un mosaico de 16x16.
    * grid  = (64/16, 64/16, N*C) = (4, 4, 96) bloques en un batch de 32:
      ejes x/y recorren el mosaico de la imagen y el eje z el par (imagen, canal).
    * Total: 4*4*96*256 = 393,216 hilos = un hilo por valor del batch (32*3*64*64).

Separar las dos etapas en dos kernels es la forma de tener una barrera global (todas
las medias listas) sin sincronizar bloques entre sí, algo que CUDA no permite dentro
de un mismo kernel.

Para comparar se incluyen tres versiones en CPU con exactamente la misma aritmética:
NumPy vectorizado, Numba compilado secuencial (1 hilo) y Numba paralelo (``prange``,
todos los núcleos lógicos).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Las ruedas pip de NVIDIA guardan NVVM y cudart en carpetas distintas; Numba espera la
# estructura de un toolkit. scripts/setup_numba.ps1 arma esa carpeta dentro del venv.
_LOCAL_CUDA_HOME = ROOT / ".venv-numba" / "cuda_home"
if "CUDA_HOME" not in os.environ and _LOCAL_CUDA_HOME.is_dir():
    os.environ["CUDA_HOME"] = str(_LOCAL_CUDA_HOME)

import math  # noqa: E402

import numpy as np  # noqa: E402
from numba import cuda, njit, prange  # noqa: E402

THREADS_PER_BLOCK_REDUCE = 256  # potencia de 2 para la reducción en árbol
TILE = 16  # mosaico 16x16 = 256 hilos por bloque en el kernel elemento a elemento


# ---------------------------------------------------------------------------
# GPU: kernels escritos a mano
# ---------------------------------------------------------------------------


@cuda.jit
def brightness_mean_kernel(images, delta, out, means):
    """Un bloque por (imagen, canal). images/out: (N, C, H, W) float32."""
    plane = cuda.blockIdx.x  # índice lineal del par (imagen, canal)
    channels = images.shape[1]
    n = plane // channels
    c = plane % channels
    height = images.shape[2]
    width = images.shape[3]
    pixels = height * width

    tid = cuda.threadIdx.x
    partial = cuda.shared.array(THREADS_PER_BLOCK_REDUCE, dtype=np.float32)

    acc = np.float32(0.0)
    d = delta[n]
    i = tid
    while i < pixels:  # bucle con paso blockDim: hilos vecinos → píxeles vecinos
        y = i // width
        x = i % width
        v = images[n, c, y, x] + d
        v = min(max(v, np.float32(0.0)), np.float32(1.0))
        out[n, c, y, x] = v
        acc += v
        i += cuda.blockDim.x
    partial[tid] = acc
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2  # reducción en árbol en memoria compartida
    while stride > 0:
        if tid < stride:
            partial[tid] += partial[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        means[n, c] = partial[0] / pixels


@cuda.jit
def contrast_kernel(out, factor, means):
    """Un hilo por píxel. grid = (W/16, H/16, N*C), block = (16, 16)."""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    plane = cuda.blockIdx.z
    channels = out.shape[1]
    n = plane // channels
    c = plane % channels
    if y < out.shape[2] and x < out.shape[3]:  # guarda por si H o W no son múltiplos de 16
        m = means[n, c]
        v = (out[n, c, y, x] - m) * factor[n] + m
        out[n, c, y, x] = min(max(v, np.float32(0.0)), np.float32(1.0))


def launch_config(shape: tuple[int, int, int, int]) -> dict[str, tuple[int, ...]]:
    """Configuración de lanzamiento para un batch (N, C, H, W)."""
    n, c, h, w = shape
    return {
        "reduce_grid": (n * c,),
        "reduce_block": (THREADS_PER_BLOCK_REDUCE,),
        "contrast_grid": (math.ceil(w / TILE), math.ceil(h / TILE), n * c),
        "contrast_block": (TILE, TILE),
    }


def photometric_gpu(d_images, d_delta, d_factor, d_out, d_means) -> None:
    """Lanza los dos kernels sobre arreglos que ya están en la GPU (sin copiar)."""
    cfg = launch_config(d_images.shape)
    brightness_mean_kernel[cfg["reduce_grid"], cfg["reduce_block"]](
        d_images, d_delta, d_out, d_means
    )
    contrast_kernel[cfg["contrast_grid"], cfg["contrast_block"]](d_out, d_factor, d_means)


def photometric_cuda(images: np.ndarray, delta: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Versión completa host→device→host, para usar desde NumPy."""
    d_images = cuda.to_device(images)
    d_delta = cuda.to_device(delta)
    d_factor = cuda.to_device(factor)
    d_out = cuda.device_array_like(d_images)
    d_means = cuda.device_array((images.shape[0], images.shape[1]), dtype=np.float32)
    photometric_gpu(d_images, d_delta, d_factor, d_out, d_means)
    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# CPU: referencias con la misma aritmética
# ---------------------------------------------------------------------------


def photometric_numpy(images: np.ndarray, delta: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """NumPy vectorizado (cada operación recorre el batch completo)."""
    out = np.clip(images + delta[:, None, None, None], 0.0, 1.0)
    means = out.mean(axis=(2, 3), keepdims=True, dtype=np.float32)
    out = (out - means) * factor[:, None, None, None] + means
    return np.clip(out, 0.0, 1.0, out=out)


def _photometric_loops(images, delta, factor, out):
    n_img, channels, height, width = images.shape
    pixels = height * width
    for plane in range(n_img * channels):  # con prange: un plano por hilo de CPU
        n = plane // channels
        c = plane % channels
        d = delta[n]
        acc = np.float32(0.0)
        for y in range(height):
            for x in range(width):
                v = min(max(images[n, c, y, x] + d, np.float32(0.0)), np.float32(1.0))
                out[n, c, y, x] = v
                acc += v
        m = acc / pixels
        f = factor[n]
        for y in range(height):
            for x in range(width):
                v = (out[n, c, y, x] - m) * f + m
                out[n, c, y, x] = min(max(v, np.float32(0.0)), np.float32(1.0))
    return out


_sequential_impl = njit(cache=True, fastmath=False)(_photometric_loops)


@njit(parallel=True, cache=True)
def _parallel_impl(images, delta, factor, out):
    n_img, channels, height, width = images.shape
    pixels = height * width
    for plane in prange(n_img * channels):
        n = plane // channels
        c = plane % channels
        d = delta[n]
        acc = np.float32(0.0)
        for y in range(height):
            for x in range(width):
                v = min(max(images[n, c, y, x] + d, np.float32(0.0)), np.float32(1.0))
                out[n, c, y, x] = v
                acc += v
        m = acc / pixels
        f = factor[n]
        for y in range(height):
            for x in range(width):
                v = (out[n, c, y, x] - m) * f + m
                out[n, c, y, x] = min(max(v, np.float32(0.0)), np.float32(1.0))
    return out


def photometric_cpu_sequential(images, delta, factor):
    """Bucles compilados con Numba, un solo hilo: la versión secuencial de referencia."""
    return _sequential_impl(images, delta, factor, np.empty_like(images))


def photometric_cpu_parallel(images, delta, factor):
    """Los mismos bucles repartidos entre los núcleos de la CPU con ``prange``."""
    return _parallel_impl(images, delta, factor, np.empty_like(images))


def sample_parameters(n: int, rng: np.random.Generator, brightness: float = 0.2,
                      contrast: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Mismos rangos que CNN2Augmentation: delta en ±0.2 y factor en [0.8, 1.2]."""
    delta = rng.uniform(-brightness, brightness, n).astype(np.float32)
    factor = rng.uniform(1.0 - contrast, 1.0 + contrast, n).astype(np.float32)
    return delta, factor
