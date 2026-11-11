"""CNN2 del Laboratorio 3 traducida de Keras a PyTorch.

La red devuelve logits. No se aplica softmax dentro de ``forward`` porque
``torch.nn.CrossEntropyLoss`` lo incorpora de forma numéricamente estable.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_SEED = 42
NUM_CLASSES = 29
EXPECTED_TRAINABLE_PARAMETERS = 97_437
EXPECTED_KERAS_TOTAL = 97_885


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Fija los RNG usados para crear el estado inicial."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CNN2(nn.Module):
    """Equivalente NCHW de ``CNN2_aug`` sin la capa de augmentation.

    El augmentation vive separado en ``src.data.asl_dataset`` para poder
    ejecutarlo después de mover el batch al dispositivo elegido.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.25)

        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout4 = nn.Dropout(p=0.50)
        self.classifier = nn.Linear(128, num_classes, bias=True)

        self.reset_keras_style_parameters()

    def reset_keras_style_parameters(self) -> None:
        """Aproxima los inicializadores Keras: Glorot, bias 0 y BN identidad."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.pool1(self.bn1(torch.relu(self.conv1(inputs)))))
        x = self.dropout2(self.pool2(self.bn2(torch.relu(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.bn3(torch.relu(self.conv3(x)))))
        x = self.global_average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout4(x)
        return self.classifier(x)


def create_cnn2(seed: int = DEFAULT_SEED, num_classes: int = NUM_CLASSES) -> CNN2:
    """Crea CNN2 en CPU con inicialización reproducible."""
    seed_everything(seed)
    return CNN2(num_classes=num_classes)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def keras_compatible_parameter_count(model: nn.Module) -> int:
    """Replica ``model.count_params()`` de Keras para esta arquitectura.

    Keras cuenta gamma/beta y también running mean/variance de BatchNorm.
    PyTorch registra estas últimas como buffers, no como ``Parameter``. Se
    excluye ``num_batches_tracked`` porque es un detalle interno de PyTorch.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            total += module.running_mean.numel() + module.running_var.numel()
    return total


def make_adam(model: nn.Module) -> torch.optim.Adam:
    """Adam equivalente al compilado en el modelo Keras guardado."""
    return torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0.0,
    )


def save_initial_checkpoint(
    output_path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    num_classes: int = NUM_CLASSES,
) -> Path:
    """Guarda en CPU el estado inicial que usarán los benchmarks."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = create_cnn2(seed=seed, num_classes=num_classes)
    payload: dict[str, Any] = {
        "architecture": "CNN2",
        "seed": seed,
        "num_classes": num_classes,
        "torch_version": str(torch.__version__),
        "trainable_parameters": trainable_parameter_count(model),
        "keras_compatible_parameters": keras_compatible_parameter_count(model),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, output)
    return output


def load_initial_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Carga el checkpoint inicial y devuelve sus metadatos."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return {key: value for key, value in payload.items() if key != "state_dict"}
