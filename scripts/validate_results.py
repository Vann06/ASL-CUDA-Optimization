"""Valida la estructura y metadatos de una ejecución de benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expect-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expect-full", action="store_true")
    args = parser.parse_args()

    error_path = args.result_dir / "error.json"
    if error_path.exists():
        error = json.loads(error_path.read_text(encoding="utf-8"))
        fail(f"La ejecución falló: {error.get('errors')}")
    payload = json.loads((args.result_dir / "benchmark.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("errors") != []:
        fail("benchmark.json no identifica una ejecución completada sin errores")
    protocol = payload["protocol"]
    config = protocol["config"]
    hardware = payload["hardware"]
    if hardware.get("selected_device") != args.expect_device:
        fail("El dispositivo registrado no coincide con el solicitado")
    for key in ("python", "torch", "torchvision", "numpy", "cuda_compiled"):
        if key not in hardware:
            fail(f"Falta hardware.{key}")
    if not protocol.get("checkpoint_sha256"):
        fail("Falta el SHA-256 del checkpoint")
    if bool(protocol.get("full_protocol")) != args.expect_full:
        fail("La marca full_protocol no coincide con lo esperado")
    if args.expect_full and not (
        config["epochs"] == 10
        and config["batch_size"] == 32
        and protocol["repetitions"] == 3
        and config["max_train_batches"] is None
        and config["max_eval_batches"] is None
    ):
        fail("Los parámetros no son el protocolo oficial")
    if "overall_process_wall_s" not in payload or not payload.get("timing_definitions"):
        fail("Faltan tiempo total o definiciones de tiempos")

    with (args.result_dir / "runs.csv").open(encoding="utf-8", newline="") as source:
        runs = list(csv.DictReader(source))
    with (args.result_dir / "epochs.csv").open(encoding="utf-8", newline="") as source:
        epochs = list(csv.DictReader(source))
    required_run = {
        "device", "seed", "epochs", "batch_size", "num_workers",
        "trainable_parameters", "keras_compatible_parameters", "training_wall_s",
        "measured_execution_s", "run_wall_s", "test_accuracy", "test_f1_macro",
        "test_transfer_s", "test_forward_s",
    }
    required_epoch = {
        "device", "repetition", "epoch", "train_load_cpu_s", "train_transfer_s",
        "train_forward_s", "train_backward_s", "train_optimizer_s",
        "train_epoch_wall_s", "val_evaluation_wall_s",
    }
    if not runs or not required_run.issubset(runs[0]):
        fail(f"runs.csv incompleto; faltan {sorted(required_run - set(runs[0] if runs else []))}")
    if not epochs or not required_epoch.issubset(epochs[0]):
        fail(f"epochs.csv incompleto; faltan {sorted(required_epoch - set(epochs[0] if epochs else []))}")
    if any(row["device"] != args.expect_device for row in runs + epochs):
        fail("Un CSV contiene un dispositivo incorrecto")
    print(f"Resultados válidos: {args.result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
