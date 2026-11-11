"""Crea el checkpoint inicial canónico de CNN2, siempre desde CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn2 import save_initial_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/initialization/cnn2_seed42_initial.pt"),
    )
    args = parser.parse_args()
    path = save_initial_checkpoint(args.output, seed=args.seed)
    print(f"Estado inicial guardado: {path}")


if __name__ == "__main__":
    main()
