"""Dataset memmap y augmentation tensorial de CNN2.

RECONSTRUCCIÓN: el módulo original no llegó al repositorio porque la regla
``**/data/`` de ``.gitignore`` también excluye ``src/data/``. Esta versión se
reescribió a partir de la especificación versionada (README, auditoría del
Lab3, ``notebooks/01_modelo_pytorch.ipynb`` y las pruebas unitarias). Respeta
la misma política, pero no garantiza igualdad muestra a muestra con el archivo
original.

Política de augmentation (solo en ``train()``; en ``eval()`` es identidad):

1. rotación uniforme en ±0.04 vueltas (±14.4°);
2. traslación uniforme en ±10 % de alto y ancho;
3. zoom uniforme en ±10 %, con el mismo factor en ambos ejes;
4. brillo: delta uniforme en ±0.2 sobre el rango ``[0,1]``;
5. contraste: factor uniforme en ``[0.8,1.2]`` alrededor de la media por canal.

Las tres operaciones geométricas son bilineales con relleno ``reflection`` y
``align_corners=False``. No hay flip horizontal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test")
IMAGE_SHAPE = (64, 64, 3)

ROTATION_FACTOR = 0.04
TRANSLATION_FACTOR = 0.1
ZOOM_FACTOR = 0.1
BRIGHTNESS_FACTOR = 0.2
CONTRAST_FACTOR = 0.2


class ASLNpyDataset(Dataset):
    """Lee ``X_<split>.npy`` / ``y_<split>.npy`` con memoria mapeada.

    Cada ``__getitem__`` copia una sola imagen, la pasa de NHWC ``uint8`` a
    NCHW ``float32`` en ``[0,1]`` y devuelve la etiqueta como ``int64``. Los
    memmaps no se serializan: cada worker los reabre al primer acceso.
    """

    def __init__(self, data_dir: str | Path, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"split debe ser uno de {SPLITS}; se recibió {split!r}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.images_path = self.data_dir / f"X_{split}.npy"
        self.labels_path = self.data_dir / f"y_{split}.npy"
        classes_path = self.data_dir / "classes.json"
        for path in (self.images_path, self.labels_path, classes_path):
            if not path.is_file():
                raise FileNotFoundError(f"No se encontró {path}")
        self.class_names: list[str] = json.loads(classes_path.read_text(encoding="utf-8"))

        self._images: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        images = self.images
        labels = self.labels
        if images.dtype != np.uint8 or images.shape[1:] != IMAGE_SHAPE:
            raise ValueError(
                f"{self.images_path} debe ser uint8 (N,64,64,3); "
                f"se encontró {images.dtype} {images.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("Imágenes y etiquetas no tienen la misma longitud")
        self._length = int(images.shape[0])

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return self._images

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            self._labels = np.load(self.labels_path, mmap_mode="r")
        return self._labels

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_images"] = None
        state["_labels"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(self.images[index], dtype=np.uint8)  # copia de una imagen
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32).div_(255.0)
        label = torch.tensor(int(self.labels[index]), dtype=torch.int64)
        return tensor.contiguous(), label


class CNN2Augmentation(nn.Module):
    """Augmentation por muestra, ejecutado en el dispositivo del batch."""

    def __init__(
        self,
        rotation: float = ROTATION_FACTOR,
        translation: float = TRANSLATION_FACTOR,
        zoom: float = ZOOM_FACTOR,
        brightness: float = BRIGHTNESS_FACTOR,
        contrast: float = CONTRAST_FACTOR,
    ) -> None:
        super().__init__()
        self.rotation = rotation
        self.translation = translation
        self.zoom = zoom
        self.brightness = brightness
        self.contrast = contrast

    @staticmethod
    def _uniform(batch: torch.Tensor, low: float, high: float, columns: int = 1) -> torch.Tensor:
        values = torch.rand(batch.shape[0], columns, device=batch.device, dtype=batch.dtype)
        return values * (high - low) + low

    @staticmethod
    def _warp(images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(theta, list(images.shape), align_corners=False)
        return F.grid_sample(
            images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return images
        batch = images.shape[0]
        zeros = images.new_zeros(batch)
        ones = images.new_ones(batch)

        # 1. Rotación alrededor del centro.
        angle = self._uniform(images, -self.rotation, self.rotation).squeeze(1) * (2.0 * math.pi)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.stack(
            (torch.stack((cos, -sin, zeros), dim=1), torch.stack((sin, cos, zeros), dim=1)),
            dim=1,
        )
        images = self._warp(images, theta)

        # 2. Traslación: una fracción f del lado equivale a 2f en coordenadas [-1,1].
        shift = self._uniform(images, -self.translation, self.translation, columns=2) * 2.0
        theta = torch.stack(
            (
                torch.stack((ones, zeros, shift[:, 0]), dim=1),
                torch.stack((zeros, ones, shift[:, 1]), dim=1),
            ),
            dim=1,
        )
        images = self._warp(images, theta)

        # 3. Zoom isotrópico (mismo factor en alto y ancho).
        scale = 1.0 + self._uniform(images, -self.zoom, self.zoom).squeeze(1)
        theta = torch.stack(
            (torch.stack((scale, zeros, zeros), dim=1), torch.stack((zeros, scale, zeros), dim=1)),
            dim=1,
        )
        images = self._warp(images, theta)

        # 4. Brillo sobre el rango [0,1].
        delta = self._uniform(images, -self.brightness, self.brightness).view(batch, 1, 1, 1)
        images = (images + delta).clamp(0.0, 1.0)

        # 5. Contraste alrededor de la media espacial de cada canal.
        factor = self._uniform(images, 1.0 - self.contrast, 1.0 + self.contrast).view(
            batch, 1, 1, 1
        )
        mean = images.mean(dim=(2, 3), keepdim=True)
        return ((images - mean) * factor + mean).clamp(0.0, 1.0)
