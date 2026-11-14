"""Benchmark del augmentation fotométrico: CPU (NumPy, Numba 1 hilo, Numba paralelo) vs
kernels CUDA escritos a mano con Numba.

Procesa una época completa del split de entrenamiento (12,180 imágenes) por batches,
con varias repeticiones y tamaños de batch. En CUDA separa:

* ``h2d_s``    copia host→device del batch y de sus parámetros;
* ``kernel_s`` los dos kernels (reducción de medias + contraste);
* ``d2h_s``    copia device→host del resultado.

Cada etapa se cierra sincronizando el stream de CUDA para que su tiempo sea real y no
asíncrono. Uso::

    .venv-numba\\Scripts\\python.exe scripts/benchmark_numba_kernel.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cuda_kernels import photometric as P  # noqa: E402

import numba  # noqa: E402
from numba import cuda  # noqa: E402

CPU_VARIANTS = {
    "cpu_numpy": P.photometric_numpy,
    "cpu_numba_1hilo": P.photometric_cpu_sequential,
    "cpu_numba_paralelo": P.photometric_cpu_parallel,
}


def load_train(data_dir: Path) -> tuple[np.ndarray, float]:
    """Fase de preparación en CPU: uint8 NHWC → float32 NCHW en [0,1]."""
    start = time.perf_counter()
    raw = np.load(data_dir / "X_train.npy", mmap_mode="r")
    images = np.empty((raw.shape[0], 3, raw.shape[1], raw.shape[2]), dtype=np.float32)
    np.divide(raw.transpose(0, 3, 1, 2), 255.0, out=images, dtype=np.float32)
    return images, time.perf_counter() - start


def run_cpu(fn, images, delta, factor, batch_size) -> dict[str, float]:
    start = time.perf_counter()
    for i in range(0, images.shape[0], batch_size):
        j = i + batch_size
        fn(images[i:j], delta[i:j], factor[i:j])
    return {"total_s": time.perf_counter() - start, "h2d_s": 0.0, "kernel_s": 0.0, "d2h_s": 0.0}


def run_cuda(images, delta, factor, batch_size, out_host) -> dict[str, float]:
    # cuda.synchronize() de Numba 0.61 falla desde la segunda llamada con el driver 610
    # (CUDA_ERROR_CONTEXT_IS_DESTROYED); todo corre en el stream por defecto, así que
    # sincronizarlo espera exactamente lo mismo.
    sync = cuda.default_stream().synchronize
    h2d = kernel = d2h = 0.0
    start = time.perf_counter()
    for i in range(0, images.shape[0], batch_size):
        j = i + batch_size
        t0 = time.perf_counter()
        d_img = cuda.to_device(images[i:j])
        d_delta = cuda.to_device(delta[i:j])
        d_factor = cuda.to_device(factor[i:j])
        d_out = cuda.device_array_like(d_img)
        d_means = cuda.device_array((d_img.shape[0], d_img.shape[1]), dtype=np.float32)
        sync()
        t1 = time.perf_counter()
        P.photometric_gpu(d_img, d_delta, d_factor, d_out, d_means)
        sync()
        t2 = time.perf_counter()
        d_out.copy_to_host(out_host[i:j])
        sync()
        t3 = time.perf_counter()
        h2d += t1 - t0
        kernel += t2 - t1
        d2h += t3 - t2
    return {"total_s": time.perf_counter() - start, "h2d_s": h2d, "kernel_s": kernel, "d2h_s": d2h}


def run_cuda_resident(d_images, d_delta, d_factor, d_out, d_means, batch_size) -> dict[str, float]:
    """Datos ya en la GPU (como en el pipeline PyTorch, donde el augmentation corre
    después de copiar el batch): solo lanzamientos de kernels sobre vistas del arreglo."""
    sync = cuda.default_stream().synchronize
    sync()
    start = time.perf_counter()
    for i in range(0, d_images.shape[0], batch_size):
        j = min(i + batch_size, d_images.shape[0])
        P.photometric_gpu(d_images[i:j], d_delta[i:j], d_factor[i:j], d_out[i:j], d_means[i:j])
    sync()
    total = time.perf_counter() - start
    return {"total_s": total, "h2d_s": 0.0, "kernel_s": total, "d2h_s": 0.0}


def gpu_info() -> dict[str, object]:
    dev = cuda.get_current_device()
    info: dict[str, object] = {
        "name": dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name),
        "compute_capability": ".".join(map(str, dev.compute_capability)),
        "multiprocessors": dev.MULTIPROCESSOR_COUNT,
        "max_threads_per_block": dev.MAX_THREADS_PER_BLOCK,
        "warp_size": dev.WARP_SIZE,
    }
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        info["driver_version"], info["memory_total"] = [s.strip() for s in smi.split(",")]
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512, 2048])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "kernel_numba")
    args = parser.parse_args()

    if not cuda.is_available():
        raise SystemExit("CUDA no disponible para Numba. Ejecute scripts/setup_numba.ps1.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparando datos en CPU...", flush=True)
    images, prep_s = load_train(args.data_dir)
    n = images.shape[0]
    delta, factor = P.sample_parameters(n, np.random.default_rng(args.seed))
    print(f"  {n} imágenes {images.shape[1:]} float32 en {prep_s:.2f} s", flush=True)

    # Verificación de equivalencia y compilación JIT (excluida de los tiempos).
    probe = slice(0, 64)
    reference = P.photometric_numpy(images[probe], delta[probe], factor[probe])
    max_abs_diff = {}
    for name, fn in {**CPU_VARIANTS, "cuda": P.photometric_cuda}.items():
        max_abs_diff[name] = float(np.abs(fn(images[probe], delta[probe], factor[probe]) - reference).max())
    print("  diferencia máxima contra NumPy:", max_abs_diff, flush=True)
    if max(max_abs_diff.values()) > 1e-5:
        raise SystemExit("Las implementaciones no coinciden.")

    out_host = np.empty_like(images)
    # Copia única de toda la época a la GPU (~600 MB de 4 GB) para la variante residente.
    d_all = cuda.to_device(images)
    d_all_delta = cuda.to_device(delta)
    d_all_factor = cuda.to_device(factor)
    d_all_out = cuda.device_array_like(d_all)
    d_all_means = cuda.device_array((n, images.shape[1]), dtype=np.float32)
    resident = (d_all, d_all_delta, d_all_factor, d_all_out, d_all_means)
    run_cuda_resident(*resident, 64)
    if float(np.abs(d_all_out[:64].copy_to_host() - reference).max()) > 1e-5:
        raise SystemExit("La variante residente no coincide con NumPy.")

    rows = []
    for bs in args.batch_sizes:
        # Warm-up por tamaño de batch: compila especializaciones y calienta la GPU.
        run_cuda(images[: bs * 4], delta[: bs * 4], factor[: bs * 4], bs, out_host)
        run_cuda_resident(*resident, bs)
        for name, fn in CPU_VARIANTS.items():
            run_cpu(fn, images[: bs * 2], delta[: bs * 2], factor[: bs * 2], bs)
        for rep in range(1, args.repetitions + 1):
            for name, fn in CPU_VARIANTS.items():
                r = run_cpu(fn, images, delta, factor, bs)
                rows.append({"variant": name, "batch_size": bs, "repetition": rep, **r})
            r = run_cuda(images, delta, factor, bs, out_host)
            rows.append({"variant": "cuda_con_copias", "batch_size": bs, "repetition": rep, **r})
            r = run_cuda_resident(*resident, bs)
            rows.append({"variant": "cuda_residente", "batch_size": bs, "repetition": rep, **r})
            last = rows[-5:]
            print(f"batch {bs:>5} rep {rep}: " + "  ".join(
                f"{x['variant']}={x['total_s']:.3f}s" for x in last), flush=True)

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cfg32 = P.launch_config((32, 3, 64, 64))
    meta = {
        "created_utc": stamp,
        "description": "Augmentation fotométrico (brillo+contraste) sobre una época de train",
        "images": n,
        "image_shape_nchw": list(images.shape[1:]),
        "prep_cpu_s": prep_s,
        "batch_sizes": args.batch_sizes,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "max_abs_diff_vs_numpy": max_abs_diff,
        "launch_config_batch32": {k: list(v) for k, v in cfg32.items()},
        "timing_definitions": {
            "total_s": "Pared de la época completa (todas las copias y kernels).",
            "h2d_s": "Copias host→device del batch y parámetros + reserva de salida.",
            "kernel_s": "brightness_mean_kernel + contrast_kernel, sincronizados.",
            "d2h_s": "Copia device→host del batch aumentado.",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "cuda_runtime": ".".join(map(str, cuda.runtime.get_version())),
            "numba_threads": numba.get_num_threads(),
        },
        "hardware": {
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "gpu": gpu_info(),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados guardados en {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
