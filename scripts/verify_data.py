"""Verifica formas, tipos, clases y SHA-256 de los artefactos ASL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    "X_train.npy": ((12_180, 64, 64, 3), "uint8"),
    "y_train.npy": ((12_180,), "int64"),
    "X_val.npy": ((2_610, 64, 64, 3), "uint8"),
    "y_val.npy": ((2_610,), "int64"),
    "X_test.npy": ((2_610, 64, 64, 3), "uint8"),
    "y_test.npy": ((2_610,), "int64"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/environment/data_manifest.json"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("evidence/asl_data_reference.json"),
        help="Manifiesto versionado contra el que se compara; use 'none' para omitir.",
    )
    args = parser.parse_args()
    artifacts = {}
    for name, (expected_shape, expected_dtype) in EXPECTED.items():
        path = args.data_dir / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape or str(array.dtype) != expected_dtype:
            raise ValueError(
                f"{name}: forma/tipo {(array.shape, str(array.dtype))}; "
                f"esperado {(expected_shape, expected_dtype)}"
            )
        artifacts[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
    classes = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))
    if len(classes) != 29 or len(set(classes)) != 29:
        raise ValueError("classes.json no contiene 29 clases únicas")
    for name in ("classes.json", "preprocess_meta.json"):
        path = args.data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Falta el artefacto requerido: {path}")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
    payload = {"classes": classes, "files": artifacts}

    reference_path = args.reference
    if str(reference_path).lower() != "none":
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        differences: list[str] = []
        if classes != reference.get("classes"):
            differences.append("El orden o contenido de classes.json no coincide")
        reference_files = reference.get("files", {})
        for name, actual in artifacts.items():
            expected = reference_files.get(name)
            if expected is None:
                differences.append(f"{name}: no existe en el manifiesto de referencia")
                continue
            for field in ("shape", "dtype", "bytes", "sha256"):
                if field in expected and actual.get(field) != expected[field]:
                    differences.append(
                        f"{name}.{field}: actual={actual.get(field)!r}; "
                        f"esperado={expected[field]!r}"
                    )
        payload["reference"] = str(reference_path.as_posix())
        payload["matches_reference"] = not differences
        payload["differences"] = differences
        if differences:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            raise ValueError(
                "Los datos no coinciden con la referencia:\n- " + "\n- ".join(differences)
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Datos verificados y equivalentes; manifiesto local: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
