"""CLI reproducible para comparar CNN2 en CPU y CUDA."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.train import TIMING_DEFINITIONS, TrainConfig, build_parser, run_repetition
from src.utils.device import select_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def nvidia_smi_snapshot() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def total_ram_bytes() -> int | None:
    if platform.system() == "Windows":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        return ",".join(versions) or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info(device: torch.device) -> dict[str, Any]:
    ram = total_ram_bytes()
    info: dict[str, Any] = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": ram,
        "ram_gib": round(ram / 1024**3, 2) if ram else None,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": None,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
    }
    try:
        import importlib.metadata

        info["torchvision"] = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        pass
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "driver_version": nvidia_driver_version(),
        }
    return info


def flatten_epoch(row: dict[str, Any], device: str) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "device": device,
        "repetition": row["repetition"],
        "epoch": row["epoch"],
        "train_loss": row["train"]["loss"],
        "train_accuracy": row["train"]["accuracy"],
        "train_examples": row["train"]["examples"],
        "train_batches": row["train"]["batches"],
        "val_loss": row["validation"]["loss"],
        "val_accuracy": row["validation"]["accuracy"],
        "val_f1_macro": row["validation"]["f1_macro"],
        "val_examples": row["validation"]["examples"],
        "val_batches": row["validation"]["batches"],
    }
    flat.update({f"train_{key}": value for key, value in row["train"]["timings"].items()})
    flat.update(
        {f"val_{key}": value for key, value in row["validation"]["timings"].items()}
    )
    return flat


def flatten_run(run: dict[str, Any]) -> dict[str, Any]:
    test = run["test"]
    metadata = run["checkpoint_metadata"]
    return {
        "repetition": run["repetition"],
        "device": run["device"],
        "seed": run["config"]["seed"],
        "epochs": run["config"]["epochs"],
        "batch_size": run["config"]["batch_size"],
        "num_workers": run["config"]["num_workers"],
        "trainable_parameters": metadata["trainable_parameters"],
        "keras_compatible_parameters": metadata["keras_compatible_parameters"],
        "data_setup_s": run["data_setup_s"],
        "warmup_s_excluded": run["warmup_s_excluded"],
        "model_restore_s": run["model_restore_s"],
        "training_wall_s": run["training_wall_s"],
        "final_evaluation_wall_s": run["final_evaluation_wall_s"],
        "measured_execution_s": run["measured_execution_s"],
        "run_wall_s": run["run_wall_s"],
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
        "test_f1_macro": test["f1_macro"],
        "test_examples": test["examples"],
        "test_batches": test["batches"],
        **{f"test_{key}": value for key, value in test["timings"].items()},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "benchmark.log", encoding="utf-8"),
        ],
        force=True,
    )


def main() -> int:
    parser = build_parser(default_repetitions=3, include_output=False)
    parser.description = __doc__
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks"))
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions debe ser positivo")

    device = select_device(args.device)  # Falla antes de crear resultados CUDA falsos.
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / f"{run_id}_{args.device}"
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_logging(output_dir)

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
    logging.info("Configuración: %s", asdict(config))
    logging.info("Repeticiones: %d", args.repetitions)
    logging.info("Checkpoint SHA-256: %s", sha256(checkpoint))

    snapshot = nvidia_smi_snapshot()
    if snapshot is not None:
        (output_dir / "nvidia-smi.txt").write_text(snapshot, encoding="utf-8")

    overall_start = time.perf_counter()
    runs: list[dict[str, Any]] = []
    try:
        for repetition in range(1, args.repetitions + 1):
            logging.info("Iniciando repetición %d/%d", repetition, args.repetitions)
            result = run_repetition(
                repetition=repetition,
                config=config,
                data_dir=args.data_dir,
                checkpoint_path=checkpoint,
            )
            runs.append(result)
            logging.info(
                "Repetición %d: test accuracy=%.6f, F1=%.6f, measured=%.3fs",
                repetition,
                result["test"]["accuracy"],
                result["test"]["f1_macro"],
                result["measured_execution_s"],
            )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "config": asdict(config),
                "repetitions": args.repetitions,
                "checkpoint": str(checkpoint.as_posix()),
                "checkpoint_sha256": sha256(checkpoint),
            },
            "hardware": hardware_info(device),
            "errors": [
                {"type": type(error).__name__, "message": str(error)}
            ],
            "completed_runs": runs,
        }
        (output_dir / "error.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logging.exception("Benchmark interrumpido; se guardó error.json")
        return 1

    payload = {
        "schema_version": 1,
        "status": "completed",
        "errors": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "config": asdict(config),
            "repetitions": args.repetitions,
            "checkpoint": str(checkpoint.as_posix()),
            "checkpoint_sha256": sha256(checkpoint),
            "full_protocol": (
                config.epochs == 10
                and config.batch_size == 32
                and args.repetitions == 3
                and config.max_train_batches is None
                and config.max_eval_batches is None
            ),
        },
        "timing_definitions": TIMING_DEFINITIONS,
        "hardware": hardware_info(device),
        "overall_process_wall_s": time.perf_counter() - overall_start,
        "runs": runs,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    epoch_rows = [
        flatten_epoch(epoch, run["device"])
        for run in runs
        for epoch in run["epochs"]
    ]
    write_csv(output_dir / "epochs.csv", epoch_rows)
    write_csv(output_dir / "runs.csv", [flatten_run(run) for run in runs])
    logging.info("Resultados guardados en %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
