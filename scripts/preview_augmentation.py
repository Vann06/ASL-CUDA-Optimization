"""Genera evidencia visual del augmentation PyTorch y comprueba eval identity."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.asl_dataset import ASLNpyDataset, CNN2Augmentation  # noqa: E402
from src.utils.device import select_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/augmentation_preview_pytorch.png"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/augmentation_validation.json")
    )
    args = parser.parse_args()

    device = select_device(args.device)
    train = ASLNpyDataset(ROOT / "data", "train")
    val = ASLNpyDataset(ROOT / "data", "val")
    test = ASLNpyDataset(ROOT / "data", "test")
    indices = [class_index * 420 for class_index in range(0, 29, 4)][:8]
    originals = torch.stack([train[index][0] for index in indices]).to(device)
    labels = [train.class_names[int(train[index][1])] for index in indices]

    augmentation = CNN2Augmentation().to(device)
    augmentation.train()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    augmented = augmentation(originals.clone())

    augmentation.eval()
    val_batch = torch.stack([val[index][0] for index in range(4)]).to(device)
    test_batch = torch.stack([test[index][0] for index in range(4)]).to(device)
    val_identical = torch.equal(augmentation(val_batch), val_batch)
    test_identical = torch.equal(augmentation(test_batch), test_batch)
    if not val_identical or not test_identical:
        raise AssertionError("El augmentation alteró validación o prueba en modo eval")

    figure, axes = plt.subplots(2, len(indices), figsize=(16, 5))
    for column, label in enumerate(labels):
        axes[0, column].imshow(originals[column].detach().cpu().permute(1, 2, 0))
        axes[1, column].imshow(augmented[column].detach().cpu().permute(1, 2, 0))
        axes[0, column].set_title(label.upper())
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Aumentada")
    figure.suptitle("CNN2 PyTorch: augmentation de entrenamiento (seed 42)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)

    report = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "seed": args.seed,
        "output": args.output.as_posix(),
        "train_original_range": [float(originals.min()), float(originals.max())],
        "train_augmented_range": [float(augmented.min()), float(augmented.max())],
        "validation_unchanged_in_eval": val_identical,
        "test_unchanged_in_eval": test_identical,
        "labels_shown": labels,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
