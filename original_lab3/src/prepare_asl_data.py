"""Prepara la submuestra reproducible usada por los modelos del Laboratorio 3.

El dataset original contiene 3,000 fotografias por clase. Para reducir el costo
se seleccionan 20 bloques de 30 imagenes por clase y se asignan bloques completos
a entrenamiento, validacion y prueba (14/3/3 bloques). De esta forma fotografias
con numeros consecutivos no quedan repartidas entre conjuntos.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SEED = 42
IMAGE_SIZE = (64, 64)
BLOCK_SIZE = 30
BLOCKS_PER_CLASS = 20
SPLIT_BLOCKS = {"train": 14, "val": 3, "test": 3}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LAB_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = LAB_DIR / "data"
DEFAULT_OUTPUT_DIR = LAB_DIR / "Notebooks" / "data"


def find_train_dir(root: Path) -> Path:
    """Localiza la carpeta que contiene las subcarpetas de las 29 clases."""
    direct_candidates = [
        root / "asl_alphabet_train",
        root / "asl_alphabet_train" / "asl_alphabet_train",
    ]
    recursive_candidates = sorted(
        (path for path in root.rglob("asl_alphabet_train") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    for path in dict.fromkeys(direct_candidates + recursive_candidates):
        if not path.exists():
            continue
        class_dirs = [child for child in path.iterdir() if child.is_dir()]
        if len(class_dirs) >= 20:
            return path
    raise FileNotFoundError(
        "No se encontro asl_alphabet_train con subcarpetas de clases dentro de "
        f"{root.resolve()}"
    )


def sequence_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"No se pudo extraer el numero de secuencia de {path}")
    return int(match.group(1))


def inventory(train_dir: Path) -> dict[str, list[Path]]:
    files_by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        label = "del" if class_dir.name.lower() == "delete" else class_dir.name.lower()
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        files_by_class[label] = paths
    return files_by_class


def build_split(files_by_class: dict[str, list[Path]]) -> dict[str, list[tuple[str, Path]]]:
    """Replica la seleccion por bloques del notebook EDA (semilla 42)."""
    rng = np.random.default_rng(SEED)
    rows: dict[str, list[tuple[str, Path]]] = {name: [] for name in SPLIT_BLOCKS}

    for label in sorted(files_by_class):
        paths = files_by_class[label]
        by_block: dict[int, list[Path]] = {}
        for path in paths:
            block_id = (sequence_number(path) - 1) // BLOCK_SIZE
            by_block.setdefault(block_id, []).append(path)

        available_blocks = np.array(sorted(by_block), dtype=np.int64)
        if len(available_blocks) < BLOCKS_PER_CLASS:
            raise ValueError(
                f"La clase {label!r} solo tiene {len(available_blocks)} bloques; "
                f"se necesitan {BLOCKS_PER_CLASS}."
            )

        selected_blocks = rng.choice(
            available_blocks, size=BLOCKS_PER_CLASS, replace=False
        )
        rng.shuffle(selected_blocks)

        start = 0
        for split_name, n_blocks in SPLIT_BLOCKS.items():
            assigned = selected_blocks[start : start + n_blocks]
            start += n_blocks
            for block_id in assigned:
                for path in by_block[int(block_id)]:
                    rows[split_name].append((label, path))

    for split_name in rows:
        rows[split_name].sort(key=lambda item: (item[0], str(item[1])))
    return rows


def load_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)


def materialize(
    rows: dict[str, list[tuple[str, Path]]], output_dir: Path, class_names: list[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_to_index = {label: index for index, label in enumerate(class_names)}

    for split_name, split_rows in rows.items():
        x = np.empty((len(split_rows), *IMAGE_SIZE, 3), dtype=np.uint8)
        y = np.empty(len(split_rows), dtype=np.int64)
        for index, (label, path) in enumerate(split_rows):
            x[index] = load_uint8(path)
            y[index] = label_to_index[label]
            if (index + 1) % 2000 == 0 or index + 1 == len(split_rows):
                print(f"{split_name}: {index + 1:,}/{len(split_rows):,} imagenes")
        np.save(output_dir / f"X_{split_name}.npy", x)
        np.save(output_dir / f"y_{split_name}.npy", y)
        print(f"Guardado {split_name}: X{x.shape}, y{y.shape}")

    (output_dir / "classes.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    metadata = {
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "block_size": BLOCK_SIZE,
        "selected_blocks_per_class": BLOCKS_PER_CLASS,
        "split_blocks": SPLIT_BLOCKS,
        "images_per_class": {
            split_name: len(split_rows) // len(class_names)
            for split_name, split_rows in rows.items()
        },
        "total_images": {name: len(split_rows) for name, split_rows in rows.items()},
        "classes": class_names,
        "preprocessing": [
            "correccion de orientacion EXIF",
            "conversion explicita a RGB",
            "redimension bilineal a 64x64",
            "almacenamiento uint8; normalizacion posterior segun el modelo",
        ],
    }
    (output_dir / "preprocess_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def validate_inventory(files_by_class: dict[str, list[Path]]) -> None:
    if len(files_by_class) != 29:
        raise ValueError(f"Se esperaban 29 clases y se encontraron {len(files_by_class)}")
    counts = {label: len(paths) for label, paths in files_by_class.items()}
    if min(counts.values()) != 3000 or max(counts.values()) != 3000:
        raise ValueError(f"Se esperaban 3,000 imagenes por clase. Conteos: {counts}")
    print("Inventario validado: 29 clases x 3,000 imagenes = 87,000 archivos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = find_train_dir(args.source_root)
    print("Dataset:", train_dir.resolve())
    files_by_class = inventory(train_dir)
    validate_inventory(files_by_class)
    rows = build_split(files_by_class)
    expected_per_class = {"train": 420, "val": 90, "test": 90}
    for split_name, expected in expected_per_class.items():
        actual = len(rows[split_name]) // len(files_by_class)
        if actual != expected:
            raise AssertionError(f"{split_name}: {actual} por clase; esperado {expected}")
    materialize(rows, args.output_dir, sorted(files_by_class))
    print("Datos preparados en:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
