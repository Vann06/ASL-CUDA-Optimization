"""Entrenamiento medible de CNN2 sin alterar su arquitectura u optimizador.

Este módulo contiene el protocolo reutilizable. ``src.benchmark`` se encarga de
persistir CSV/JSON y de repetir ejecuciones. Los límites ``max_*_batches`` son
solo para smoke tests; el benchmark oficial debe dejarlos en ``None``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.data.asl_dataset import ASLNpyDataset, CNN2Augmentation
from src.models.cnn2 import create_cnn2, load_initial_checkpoint, make_adam, seed_everything
from src.utils.device import select_device


T = TypeVar("T")
NUM_CLASSES = 29


@dataclass(frozen=True)
class TrainConfig:
    device: str
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    warmup_batches: int = 2
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device debe ser 'cpu' o 'cuda'")
        for name in ("epochs", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        for name in ("num_workers", "warmup_batches"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} no puede ser negativo")


TIMING_DEFINITIONS = {
    "load_cpu_s": (
        "Espera observada al solicitar el siguiente batch: memmap, copia por muestra, "
        "NHWC→NCHW, normalización y collate. Con workers puede excluir trabajo prefetched."
    ),
    "transfer_s": "Copia de imágenes y etiquetas al dispositivo; sincronizada en CUDA.",
    "augmentation_s": "Cinco transformaciones de entrenamiento sobre el dispositivo.",
    "zero_grad_s": "optimizer.zero_grad(set_to_none=True).",
    "forward_s": "Solo CNN2(inputs); no incluye loss.",
    "loss_s": "CrossEntropyLoss sobre logits y etiquetas.",
    "backward_s": "Solo loss.backward().",
    "optimizer_s": "Solo optimizer.step().",
    "metrics_s": "Argmax, conteos y transferencias necesarias para métricas.",
    "epoch_wall_s": "Pared total de la época; contiene todos los componentes y overhead.",
    "evaluation_wall_s": "Pared total de una evaluación completa.",
    "training_wall_s": (
        "Pared del protocolo de épocas; incluye entrenamiento y la validación posterior "
        "a cada época, pero no la evaluación final de test."
    ),
    "measured_execution_s": (
        "Restauración del modelo + entrenamiento fijo + evaluación final; excluye warm-up."
    ),
    "run_wall_s": "Pared de la repetición completa, incluyendo setup y warm-up.",
}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation: Callable[[], T]) -> tuple[T, float]:
    _synchronize(device)
    start = time.perf_counter()
    result = operation()
    _synchronize(device)
    return result, time.perf_counter() - start


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: ASLNpyDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_seed if num_workers > 0 else None,
        generator=generator,
    )


def _empty_timings() -> dict[str, float]:
    return {
        "load_cpu_s": 0.0,
        "transfer_s": 0.0,
        "augmentation_s": 0.0,
        "zero_grad_s": 0.0,
        "forward_s": 0.0,
        "loss_s": 0.0,
        "backward_s": 0.0,
        "optimizer_s": 0.0,
        "metrics_s": 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    augmentation: CNN2Augmentation,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.train()
    augmentation.train()
    timings = _empty_timings()
    loss_sum = 0.0
    correct = 0
    examples = 0
    batches = 0
    epoch_start = time.perf_counter()
    iterator_start = time.perf_counter()
    iterator = iter(loader)
    timings["load_cpu_s"] += time.perf_counter() - iterator_start

    while max_batches is None or batches < max_batches:
        load_start = time.perf_counter()
        try:
            cpu_images, cpu_labels = next(iterator)
        except StopIteration:
            break
        timings["load_cpu_s"] += time.perf_counter() - load_start

        (images, labels), elapsed = _timed(
            device,
            lambda: (
                cpu_images.to(device, non_blocking=device.type == "cuda"),
                cpu_labels.to(device, non_blocking=device.type == "cuda"),
            ),
        )
        timings["transfer_s"] += elapsed

        images, elapsed = _timed(device, lambda: augmentation(images))
        timings["augmentation_s"] += elapsed

        _, elapsed = _timed(device, lambda: optimizer.zero_grad(set_to_none=True))
        timings["zero_grad_s"] += elapsed
        logits, elapsed = _timed(device, lambda: model(images))
        timings["forward_s"] += elapsed
        loss, elapsed = _timed(device, lambda: criterion(logits, labels))
        timings["loss_s"] += elapsed
        _, elapsed = _timed(device, loss.backward)
        timings["backward_s"] += elapsed
        _, elapsed = _timed(device, optimizer.step)
        timings["optimizer_s"] += elapsed

        def update_metrics() -> tuple[float, int]:
            predictions = logits.detach().argmax(dim=1)
            return float(loss.detach().item()), int((predictions == labels).sum().item())

        (loss_value, batch_correct), elapsed = _timed(device, update_metrics)
        timings["metrics_s"] += elapsed
        batch_size = labels.shape[0]
        loss_sum += loss_value * batch_size
        correct += batch_correct
        examples += batch_size
        batches += 1

    epoch_wall = time.perf_counter() - epoch_start
    component_sum = sum(timings.values())
    if examples == 0:
        raise RuntimeError("El DataLoader de entrenamiento no produjo batches")
    return {
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
        "examples": examples,
        "batches": batches,
        "timings": {
            **timings,
            "component_sum_s": component_sum,
            "epoch_wall_s": epoch_wall,
            "unattributed_overhead_s": epoch_wall - component_sum,
        },
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    timings = {
        "load_cpu_s": 0.0,
        "transfer_s": 0.0,
        "forward_s": 0.0,
        "loss_s": 0.0,
        "metrics_s": 0.0,
    }
    loss_sum = 0.0
    examples = 0
    batches = 0
    predictions_cpu: list[torch.Tensor] = []
    labels_cpu: list[torch.Tensor] = []
    evaluation_start = time.perf_counter()
    iterator_start = time.perf_counter()
    iterator = iter(loader)
    timings["load_cpu_s"] += time.perf_counter() - iterator_start

    with torch.no_grad():
        while max_batches is None or batches < max_batches:
            load_start = time.perf_counter()
            try:
                cpu_images, cpu_labels = next(iterator)
            except StopIteration:
                break
            timings["load_cpu_s"] += time.perf_counter() - load_start

            (images, labels), elapsed = _timed(
                device,
                lambda: (
                    cpu_images.to(device, non_blocking=device.type == "cuda"),
                    cpu_labels.to(device, non_blocking=device.type == "cuda"),
                ),
            )
            timings["transfer_s"] += elapsed
            logits, elapsed = _timed(device, lambda: model(images))
            timings["forward_s"] += elapsed
            loss, elapsed = _timed(device, lambda: criterion(logits, labels))
            timings["loss_s"] += elapsed

            def collect_metrics() -> tuple[float, torch.Tensor, torch.Tensor]:
                return (
                    float(loss.item()),
                    logits.argmax(dim=1).cpu(),
                    labels.cpu(),
                )

            (loss_value, prediction_batch, label_batch), elapsed = _timed(
                device, collect_metrics
            )
            timings["metrics_s"] += elapsed
            batch_size = labels.shape[0]
            loss_sum += loss_value * batch_size
            examples += batch_size
            predictions_cpu.append(prediction_batch)
            labels_cpu.append(label_batch)
            batches += 1

    if examples == 0:
        raise RuntimeError("El DataLoader de evaluación no produjo batches")
    metric_start = time.perf_counter()
    predictions = torch.cat(predictions_cpu).numpy()
    targets = torch.cat(labels_cpu).numpy()
    accuracy = float((predictions == targets).mean())
    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )
    timings["metrics_s"] += time.perf_counter() - metric_start
    evaluation_wall = time.perf_counter() - evaluation_start
    component_sum = sum(timings.values())
    return {
        "loss": loss_sum / examples,
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "examples": examples,
        "batches": batches,
        "timings": {
            **timings,
            "component_sum_s": component_sum,
            "evaluation_wall_s": evaluation_wall,
            "unattributed_overhead_s": evaluation_wall - component_sum,
        },
    }


def warm_up(
    dataset: ASLNpyDataset,
    checkpoint_path: Path,
    config: TrainConfig,
    device: torch.device,
) -> float:
    if config.warmup_batches == 0:
        return 0.0
    start = time.perf_counter()
    seed_everything(config.seed)
    loader = make_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        device=device,
    )
    model = create_cnn2(seed=config.seed).to(device)
    load_initial_checkpoint(model, checkpoint_path, map_location=device)
    augmentation = CNN2Augmentation().to(device)
    optimizer = make_adam(model)
    criterion = nn.CrossEntropyLoss()
    train_one_epoch(
        model,
        augmentation,
        loader,
        optimizer,
        criterion,
        device,
        max_batches=config.warmup_batches,
    )
    _synchronize(device)
    return time.perf_counter() - start


def run_repetition(
    *,
    repetition: int,
    config: TrainConfig,
    data_dir: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    config.validate()
    device = select_device(config.device)
    data_path = Path(data_dir)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint inicial: {checkpoint}")

    run_start = time.perf_counter()
    setup_start = time.perf_counter()
    train_dataset = ASLNpyDataset(data_path, "train")
    val_dataset = ASLNpyDataset(data_path, "val")
    test_dataset = ASLNpyDataset(data_path, "test")
    data_setup_s = time.perf_counter() - setup_start

    warmup_s = warm_up(train_dataset, checkpoint, config, device)

    # El warm-up se descarta por completo. Se restauran RNG, loaders, pesos y Adam.
    seed_everything(config.seed)
    measured_loader_setup_start = time.perf_counter()
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        device=device,
    )
    data_setup_s += time.perf_counter() - measured_loader_setup_start

    measured_start = time.perf_counter()
    restore_start = time.perf_counter()
    model = create_cnn2(seed=config.seed).to(device)
    checkpoint_metadata = load_initial_checkpoint(model, checkpoint, map_location=device)
    augmentation = CNN2Augmentation().to(device)
    optimizer = make_adam(model)
    criterion = nn.CrossEntropyLoss()
    _synchronize(device)
    model_restore_s = time.perf_counter() - restore_start

    epoch_rows: list[dict[str, Any]] = []
    training_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_result = train_one_epoch(
            model,
            augmentation,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=config.max_train_batches,
        )
        validation_result = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=config.max_eval_batches,
        )
        epoch_rows.append(
            {
                "repetition": repetition,
                "epoch": epoch,
                "train": train_result,
                "validation": validation_result,
            }
        )
    training_wall_s = time.perf_counter() - training_start

    test_result = evaluate(
        model,
        test_loader,
        criterion,
        device,
        max_batches=config.max_eval_batches,
    )
    measured_execution_s = time.perf_counter() - measured_start
    return {
        "repetition": repetition,
        "config": asdict(config),
        "device": str(device),
        "checkpoint_metadata": checkpoint_metadata,
        "data_setup_s": data_setup_s,
        "warmup_s_excluded": warmup_s,
        "model_restore_s": model_restore_s,
        "training_wall_s": training_wall_s,
        "final_evaluation_wall_s": test_result["timings"]["evaluation_wall_s"],
        "measured_execution_s": measured_execution_s,
        "run_wall_s": time.perf_counter() - run_start,
        "epochs": epoch_rows,
        "test": test_result,
    }


def build_parser(
    default_repetitions: int = 1,
    *,
    include_output: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=default_repetitions)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/initialization/cnn2_seed42_initial.pt"),
    )
    if include_output:
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("results/train_quick.json"),
        )
    return parser


def main() -> int:
    args = build_parser(default_repetitions=1).parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions debe ser positivo")
    config = TrainConfig(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        warmup_batches=args.warmup_batches,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    results = [
        run_repetition(
            repetition=index,
            config=config,
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
        )
        for index in range(1, args.repetitions + 1)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"timing_definitions": TIMING_DEFINITIONS, "runs": results},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Resultado guardado en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
