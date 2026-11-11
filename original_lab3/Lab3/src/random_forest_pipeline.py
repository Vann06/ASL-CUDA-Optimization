"""Entrena y compara Random Forest para el reconocimiento de ASL Alphabet.

La representacion clasica combina HOG (bordes y orientacion local de los dedos)
con promedios e histogramas de color. La configuracion se selecciona usando
validacion; el conjunto de prueba se consulta una sola vez para el ganador.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = LAB_DIR / "Notebooks"
DATA_DIR = NOTEBOOKS_DIR / "data"
RESULTS_DIR = NOTEBOOKS_DIR / "results"
MODELS_DIR = NOTEBOOKS_DIR / "models"
IMAGE_SIZE = (64, 64)
OWN_PHOTO_GROUPS = {
    "Integrante A": {"i", "j", "k", "l", "r"},
    "Integrante B": {"m", "n", "s", "u", "x"},
    "Integrante C": {"o", "v", "w", "y", "z"},
}


RF_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate": "RF_100_depth20_sqrt",
        "n_estimators": 100,
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    {
        "candidate": "RF_200_full_sqrt",
        "n_estimators": 200,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    {
        "candidate": "RF_300_depth30_sqrt",
        "n_estimators": 300,
        "max_depth": 30,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    {
        "candidate": "RF_300_full_log2_leaf2",
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "log2",
        "min_samples_leaf": 2,
    },
]


def load_arrays(data_dir: Path = DATA_DIR) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays = {
        "X_train": np.load(data_dir / "X_train.npy", mmap_mode="r"),
        "y_train": np.load(data_dir / "y_train.npy"),
        "X_val": np.load(data_dir / "X_val.npy", mmap_mode="r"),
        "y_val": np.load(data_dir / "y_val.npy"),
        "X_test": np.load(data_dir / "X_test.npy", mmap_mode="r"),
        "y_test": np.load(data_dir / "y_test.npy"),
    }
    class_names = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    return arrays, class_names


def hog_color_features(image: np.ndarray) -> np.ndarray:
    """Calcula HOG 9-bin/8px/2x2 y un resumen espacial/cromatico."""
    rgb = image.astype(np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gy[0, :] = gray[1, :] - gray[0, :]
    gy[-1, :] = gray[-1, :] - gray[-2, :]

    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0) / 20.0
    lower = np.floor(angle).astype(np.int16) % 9
    upper = (lower + 1) % 9
    upper_weight = angle - np.floor(angle)
    lower_weight = 1.0 - upper_weight

    cell_hist = np.empty((8, 8, 9), dtype=np.float32)
    for bin_index in range(9):
        votes = magnitude * (
            (lower == bin_index) * lower_weight + (upper == bin_index) * upper_weight
        )
        cell_hist[:, :, bin_index] = votes.reshape(8, 8, 8, 8).sum(axis=(1, 3))

    blocks = np.concatenate(
        [
            cell_hist[:-1, :-1],
            cell_hist[:-1, 1:],
            cell_hist[1:, :-1],
            cell_hist[1:, 1:],
        ],
        axis=2,
    )
    blocks /= np.sqrt(np.sum(blocks**2, axis=2, keepdims=True) + 1e-6)
    hog = blocks.ravel()

    color_cells = rgb.reshape(8, 8, 8, 8, 3).mean(axis=(1, 3)).ravel()
    color_hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0))
        color_hist_parts.append(hist.astype(np.float32) / rgb[:, :, channel].size)
    color_hist = np.concatenate(color_hist_parts)
    return np.concatenate([hog, color_cells, color_hist]).astype(np.float32)


def extract_feature_matrix(images: np.ndarray, name: str) -> np.ndarray:
    features = np.empty((len(images), 2004), dtype=np.float32)
    for index, image in enumerate(images):
        features[index] = hog_color_features(image)
        if (index + 1) % 2000 == 0 or index + 1 == len(images):
            print(f"Caracteristicas {name}: {index + 1:,}/{len(images):,}")
    return features


def load_or_create_features(
    arrays: dict[str, np.ndarray], data_dir: Path = DATA_DIR
) -> dict[str, np.ndarray]:
    feature_sets: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        cache_path = data_dir / f"F_{split_name}_hog_color.npy"
        if cache_path.exists():
            feature_sets[split_name] = np.load(cache_path, mmap_mode="r")
            print(f"Cache reutilizada: {cache_path.name} {feature_sets[split_name].shape}")
            continue
        features = extract_feature_matrix(arrays[f"X_{split_name}"], split_name)
        np.save(cache_path, features)
        feature_sets[split_name] = np.load(cache_path, mmap_mode="r")
        print(f"Cache guardada: {cache_path.name} {features.shape}")
    return feature_sets


def train_candidates(
    features: dict[str, np.ndarray], arrays: dict[str, np.ndarray]
) -> tuple[RandomForestClassifier, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_model: RandomForestClassifier | None = None
    best_key = (-np.inf, -np.inf)

    for config in RF_CANDIDATES:
        print(f"\nEntrenando {config['candidate']}...")
        start = time.perf_counter()
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            max_features=config["max_features"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=SEED,
            n_jobs=-1,
            class_weight=None,
        )
        model.fit(features["train"], arrays["y_train"])
        elapsed = time.perf_counter() - start
        train_pred = model.predict(features["train"])
        val_pred = model.predict(features["val"])
        row = {
            **config,
            "accuracy_train": accuracy_score(arrays["y_train"], train_pred),
            "accuracy_val": accuracy_score(arrays["y_val"], val_pred),
            "f1_macro_val": f1_score(arrays["y_val"], val_pred, average="macro"),
            "tiempo_entreno_s": round(elapsed, 2),
        }
        rows.append(row)
        print(
            f"{config['candidate']}: val_acc={row['accuracy_val']:.4f}, "
            f"val_f1={row['f1_macro_val']:.4f}, tiempo={elapsed:.1f}s"
        )
        key = (row["accuracy_val"], row["f1_macro_val"])
        if key > best_key:
            best_key = key
            best_model = model

    if best_model is None:
        raise RuntimeError("No se entreno ningun candidato de Random Forest")
    candidates_df = pd.DataFrame(rows).sort_values(
        ["accuracy_val", "f1_macro_val"], ascending=False
    )
    return best_model, candidates_df


def save_rf_results(
    model: RandomForestClassifier,
    candidates_df: pd.DataFrame,
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    class_names: list[str],
    results_dir: Path = RESULTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> tuple[np.ndarray, dict[str, Any]]:
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(results_dir / "rf_candidates.csv", index=False)

    y_pred = model.predict(features["test"])
    report = classification_report(
        arrays["y_test"],
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(results_dir / "classification_report_rf.csv")

    cm = confusion_matrix(arrays["y_test"], y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        results_dir / "confusion_matrix_rf.csv"
    )
    normalized_cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized_cm, cmap="mako" if "mako" in plt.colormaps() else "Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=90)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Random Forest - matriz de confusion normalizada por clase")
    fig.colorbar(image, ax=ax, label="Proporcion por fila")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix_rf.png", dpi=180)
    plt.close(fig)

    best_row = candidates_df.iloc[0].to_dict()
    metrics = {
        "modelo": "RandomForest_HOG_color",
        "algoritmo": "RandomForestClassifier",
        "representacion": "HOG 9 bins, celdas 8x8, bloques 2x2 + color",
        "n_features": int(features["train"].shape[1]),
        "best_candidate": best_row["candidate"],
        "best_params": {
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": None
            if pd.isna(best_row["max_depth"])
            else int(best_row["max_depth"]),
            "max_features": best_row["max_features"],
            "min_samples_leaf": int(best_row["min_samples_leaf"]),
        },
        "accuracy_val": float(best_row["accuracy_val"]),
        "f1_macro_val": float(best_row["f1_macro_val"]),
        "accuracy_test": float(accuracy_score(arrays["y_test"], y_pred)),
        "balanced_accuracy_test": float(
            balanced_accuracy_score(arrays["y_test"], y_pred)
        ),
        "f1_macro_test": float(
            f1_score(arrays["y_test"], y_pred, average="macro")
        ),
        "tiempo_entreno_s": float(best_row["tiempo_entreno_s"]),
        "seed": SEED,
    }
    (results_dir / "metrics_random_forest.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(model, models_dir / "random_forest_hog_color.joblib", compress=3)
    print("\nMetricas de prueba Random Forest:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return y_pred, metrics


def load_external_images(paths: list[Path]) -> np.ndarray:
    images = np.empty((len(paths), *IMAGE_SIZE, 3), dtype=np.uint8)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            images[index] = np.asarray(
                image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8
            )
    return images


def keras_predictions(
    model_path: Path, images_uint8: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = load_keras_compat(model_path)
    probabilities = model.predict(images_uint8.astype(np.float32) / 255.0, verbose=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return predictions, confidence


@lru_cache(maxsize=2)
def load_keras_compat(model_path: Path):
    """Carga modelos Keras 3.15 desde el entorno Python 3.10/Keras 3.12.

    Keras 3.15 agrego campos de configuracion que 3.12 no reconoce. Se crea una
    copia temporal sin esos campos; los pesos y el modelo original no se alteran.
    """
    from tensorflow import keras

    drop_keys = {
        "input_axes",
        "output_axes",
        "quantization_config",
        "renorm",
        "renorm_clipping",
        "renorm_momentum",
    }

    def clean_config(value):
        if isinstance(value, dict):
            return {
                key: clean_config(item)
                for key, item in value.items()
                if key not in drop_keys
            }
        if isinstance(value, list):
            return [clean_config(item) for item in value]
        return value

    compatible_path = DATA_DIR / f"compat_{model_path.name}"
    with ZipFile(model_path) as source, ZipFile(
        compatible_path, "w", ZIP_DEFLATED
    ) as destination:
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename == "config.json":
                content = json.dumps(
                    clean_config(json.loads(content)), ensure_ascii=False
                ).encode("utf-8")
            destination.writestr(item, content)
    model = keras.models.load_model(compatible_path, compile=False)
    print(
        f"Modelo compatible cargado: {model_path.name} -> {model.name}, "
        f"{model.count_params():,} parametros"
    )
    return model


def compare_main_models(
    rf_model: RandomForestClassifier,
    rf_test_pred: np.ndarray,
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    class_names: list[str],
    rf_metrics: dict[str, Any],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    normal_path = NOTEBOOKS_DIR / "mejor_modelo.keras"
    aug_path = NOTEBOOKS_DIR / "mejor_modelo_aug.keras"
    normal_pred, _ = keras_predictions(normal_path, arrays["X_test"])
    aug_pred, _ = keras_predictions(aug_path, arrays["X_test"])

    model_predictions = {
        "CNN2_regularizada": normal_pred,
        "CNN2_aug": aug_pred,
        "RandomForest_HOG_color": rf_test_pred,
    }
    rows = []
    for model_name, prediction in model_predictions.items():
        rows.append(
            {
                "modelo": model_name,
                "familia": "Random Forest" if model_name.startswith("Random") else "CNN",
                "augmentation": model_name.endswith("_aug"),
                "accuracy_test_recalculada": accuracy_score(arrays["y_test"], prediction),
                "f1_macro_test_recalculado": f1_score(
                    arrays["y_test"], prediction, average="macro"
                ),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        "accuracy_test_recalculada", ascending=False
    )
    comparison.to_csv(results_dir / "comparacion_principal.csv", index=False)

    focus_labels = ["a", "b", "c", "d", "e"]
    focus_indices = np.array([class_names.index(label) for label in focus_labels])
    focus_mask = np.isin(arrays["y_test"], focus_indices)
    focus_rows = []
    for model_name, prediction in model_predictions.items():
        focus_rows.append(
            {
                "modelo": model_name,
                "n_imagenes_A_E": int(focus_mask.sum()),
                "accuracy_A_E": accuracy_score(
                    arrays["y_test"][focus_mask], prediction[focus_mask]
                ),
                "f1_macro_A_E": f1_score(
                    arrays["y_test"][focus_mask],
                    prediction[focus_mask],
                    labels=focus_indices,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    pd.DataFrame(focus_rows).sort_values("accuracy_A_E", ascending=False).to_csv(
        results_dir / "comparacion_A_E_interna.csv", index=False
    )
    return comparison


def predict_official_a_to_e(
    rf_model: RandomForestClassifier,
    class_names: list[str],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    official_dir = LAB_DIR / "data" / "asl_alphabet_test" / "asl_alphabet_test"
    paths = [official_dir / f"{letter}_test.jpg" for letter in "ABCDE"]
    images = load_external_images(paths)
    features = np.stack([hog_color_features(image) for image in images])
    rf_probabilities = rf_model.predict_proba(features)
    rf_pred = rf_probabilities.argmax(axis=1)
    normal_pred, normal_conf = keras_predictions(
        NOTEBOOKS_DIR / "mejor_modelo.keras", images
    )
    aug_pred, aug_conf = keras_predictions(
        NOTEBOOKS_DIR / "mejor_modelo_aug.keras", images
    )

    rows = []
    for index, path in enumerate(paths):
        expected = path.stem.split("_")[0].lower()
        rows.append(
            {
                "archivo": path.name,
                "etiqueta_esperada": expected,
                "CNN2_normal_pred": class_names[int(normal_pred[index])],
                "CNN2_normal_confianza": float(normal_conf[index]),
                "CNN2_aug_pred": class_names[int(aug_pred[index])],
                "CNN2_aug_confianza": float(aug_conf[index]),
                "RandomForest_pred": class_names[int(rf_pred[index])],
                "RandomForest_confianza": float(rf_probabilities[index].max()),
            }
        )
    predictions = pd.DataFrame(rows)
    for prefix in ("CNN2_normal", "CNN2_aug", "RandomForest"):
        predictions[f"{prefix}_correcto"] = (
            predictions[f"{prefix}_pred"] == predictions["etiqueta_esperada"]
        )
    predictions.to_csv(results_dir / "predicciones_oficiales_A_E.csv", index=False)
    return predictions


def predict_own_photos(
    rf_model: RandomForestClassifier,
    class_names: list[str],
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    own_root = LAB_DIR / "asl_alphabet_test"
    paths = sorted(
        path
        for path in own_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        print("No se encontraron fotos propias organizadas en subcarpetas por letra.")
        return pd.DataFrame(), pd.DataFrame()

    images = load_external_images(paths)
    feature_matrix = np.stack([hog_color_features(image) for image in images])
    rf_probabilities = rf_model.predict_proba(feature_matrix)
    rf_pred = rf_model.classes_[rf_probabilities.argmax(axis=1)]
    normal_pred, normal_conf = keras_predictions(
        NOTEBOOKS_DIR / "mejor_modelo.keras", images
    )
    aug_pred, aug_conf = keras_predictions(
        NOTEBOOKS_DIR / "mejor_modelo_aug.keras", images
    )

    rows = []
    for index, path in enumerate(paths):
        expected = path.parent.name.lower()
        contributor = next(
            (
                name
                for name, labels in OWN_PHOTO_GROUPS.items()
                if expected in labels
            ),
            "Sin asignar",
        )
        rows.append(
            {
                "archivo": str(path.relative_to(LAB_DIR)),
                "integrante": contributor,
                "etiqueta_esperada": expected,
                "CNN2_normal_pred": class_names[int(normal_pred[index])],
                "CNN2_normal_confianza": float(normal_conf[index]),
                "CNN2_aug_pred": class_names[int(aug_pred[index])],
                "CNN2_aug_confianza": float(aug_conf[index]),
                "RandomForest_pred": class_names[int(rf_pred[index])],
                "RandomForest_confianza": float(rf_probabilities[index].max()),
            }
        )
    predictions = pd.DataFrame(rows)
    summary_rows = []
    for prefix in ("CNN2_normal", "CNN2_aug", "RandomForest"):
        correct_column = f"{prefix}_correcto"
        predictions[correct_column] = (
            predictions[f"{prefix}_pred"] == predictions["etiqueta_esperada"]
        )
        summary_rows.append(
            {
                "modelo": prefix,
                "n_fotos": len(predictions),
                "n_letras": predictions["etiqueta_esperada"].nunique(),
                "accuracy_fotos_propias": predictions[correct_column].mean(),
                "confianza_media": predictions[f"{prefix}_confianza"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "accuracy_fotos_propias", ascending=False
    )
    predictions.to_csv(results_dir / "predicciones_fotos_propias.csv", index=False)
    summary.to_csv(results_dir / "comparacion_fotos_propias.csv", index=False)

    contributor_rows = []
    for contributor, group in predictions.groupby("integrante"):
        for prefix in ("CNN2_normal", "CNN2_aug", "RandomForest"):
            contributor_rows.append(
                {
                    "integrante": contributor,
                    "modelo": prefix,
                    "n_fotos": len(group),
                    "n_letras": group["etiqueta_esperada"].nunique(),
                    "letras": ", ".join(
                        sorted(group["etiqueta_esperada"].str.upper().unique())
                    ),
                    "accuracy": group[f"{prefix}_correcto"].mean(),
                    "confianza_media": group[f"{prefix}_confianza"].mean(),
                }
            )
    contributor_summary = pd.DataFrame(contributor_rows).sort_values(
        ["integrante", "accuracy"], ascending=[True, False]
    )
    contributor_summary.to_csv(
        results_dir / "comparacion_fotos_propias_por_integrante.csv", index=False
    )

    own_cm = pd.crosstab(
        predictions["etiqueta_esperada"], predictions["CNN2_aug_pred"]
    )
    own_cm.to_csv(results_dir / "confusion_fotos_propias_cnn_aug.csv")
    fig_width = max(10, 0.9 * len(own_cm.columns) + 4)
    fig_height = max(6, 0.55 * len(own_cm.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    matrix_image = ax.imshow(own_cm.to_numpy(), cmap="Blues")
    ax.set_xticks(np.arange(len(own_cm.columns)), labels=own_cm.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(own_cm.index)), labels=own_cm.index)
    ax.set_xlabel("Predicho por CNN2 con augmentation")
    ax.set_ylabel("Real")
    ax.set_title("Fotos propias - matriz de confusion")
    for row_index in range(own_cm.shape[0]):
        for column_index in range(own_cm.shape[1]):
            value = int(own_cm.iat[row_index, column_index])
            if value:
                ax.text(column_index, row_index, value, ha="center", va="center")
    fig.colorbar(matrix_image, ax=ax, label="Numero de fotos")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_fotos_propias_cnn_aug.png", dpi=180)
    plt.close(fig)

    labels = sorted(predictions["etiqueta_esperada"].unique())
    columns = 5
    rows_count = int(np.ceil(len(labels) / columns))
    fig, axes = plt.subplots(
        rows_count, columns, figsize=(3 * columns, 3.2 * rows_count), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for position, label in enumerate(labels):
        example_index = predictions.index[
            predictions["etiqueta_esperada"] == label
        ][0]
        prediction = predictions.loc[example_index]
        ax = axes.flat[position]
        ax.imshow(images[example_index])
        ax.set_title(
            f"Real {label.upper()} -> {prediction['CNN2_aug_pred'].upper()}\n"
            f"conf. {prediction['CNN2_aug_confianza']:.1%}"
        )
        ax.axis("off")
        color = "#2ca02c" if prediction["CNN2_aug_correcto"] else "#d62728"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    fig.suptitle("Ejemplos de fotos propias evaluadas por CNN2 con augmentation")
    fig.tight_layout()
    fig.savefig(results_dir / "ejemplos_fotos_propias.png", dpi=180)
    plt.close(fig)

    by_class_rows = []
    for label, group in predictions.groupby("etiqueta_esperada"):
        for prefix in ("CNN2_normal", "CNN2_aug", "RandomForest"):
            by_class_rows.append(
                {
                    "letra": label,
                    "modelo": prefix,
                    "n_fotos": len(group),
                    "recall": group[f"{prefix}_correcto"].mean(),
                    "confianza_media": group[f"{prefix}_confianza"].mean(),
                }
            )
    pd.DataFrame(by_class_rows).to_csv(
        results_dir / "fotos_propias_por_letra.csv", index=False
    )
    return predictions, summary


def main() -> None:
    arrays, class_names = load_arrays()
    print("Clases:", class_names)
    print(
        "Particiones:",
        arrays["X_train"].shape,
        arrays["X_val"].shape,
        arrays["X_test"].shape,
    )
    features = load_or_create_features(arrays)
    saved_model_path = MODELS_DIR / "random_forest_hog_color.joblib"
    saved_metrics_path = RESULTS_DIR / "metrics_random_forest.json"
    saved_candidates_path = RESULTS_DIR / "rf_candidates.csv"
    if saved_model_path.exists() and saved_metrics_path.exists() and saved_candidates_path.exists():
        print("Reutilizando Random Forest ya entrenado:", saved_model_path)
        model = joblib.load(saved_model_path)
        metrics = json.loads(saved_metrics_path.read_text(encoding="utf-8"))
        rf_test_pred = model.predict(features["test"])
    else:
        model, candidates_df = train_candidates(features, arrays)
        rf_test_pred, metrics = save_rf_results(
            model, candidates_df, features, arrays, class_names
        )
    comparison = compare_main_models(
        model, rf_test_pred, features, arrays, class_names, metrics
    )
    official_predictions = predict_official_a_to_e(model, class_names)
    own_predictions, own_summary = predict_own_photos(model, class_names)
    print("\nComparacion principal:")
    print(comparison.to_string(index=False))
    print("\nPrueba tecnica A-E con imagenes oficiales de Kaggle:")
    print(official_predictions.to_string(index=False))
    if not own_predictions.empty:
        print("\nPrueba con fotos propias disponibles:")
        print(own_summary.to_string(index=False))


if __name__ == "__main__":
    main()
