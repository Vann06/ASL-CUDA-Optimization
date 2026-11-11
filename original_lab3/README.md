# Laboratorio 3 - Reconocimiento de ASL

Este laboratorio tuvo como objetivo clasificar las 29 clases
del dataset ASL Alphabet. El trabajo usa 600 imágenes por clase (17,400 en total),
separadas por bloques en entrenamiento, validación y prueba para reducir la fuga
entre fotografías consecutivas.

## Qué contiene cada archivo

| Requisito | Evidencia principal |
|---|---|
| EDA, ejemplos, balance, formato y preprocesamiento | `Notebooks/0.Laboratorio_3_EDA_ASL.ipynb` |
| Dos CNN, fully-connected y augmentation | `Notebooks/01_modelos_dl.ipynb` |
| Random Forest, comparación, A-E, fotos propias y accesibilidad | `Notebooks/02_random_forest_comparacion_final.ipynb` |
| Preparación reproducible de los datos | `src/prepare_asl_data.py` |
| Entrenamiento y evaluación reproducible | `src/random_forest_pipeline.py` |
| Resultados tabulares y matrices | `Notebooks/results/` |
| Mejor Random Forest exportado | `Notebooks/models/random_forest_hog_color.joblib` |

El informe narrativo consolidado también está disponible en
`INFORME_FINAL_LAB3.md`.

## Resultados principales

| Modelo | Accuracy prueba | F1 macro prueba |
|---|---:|---:|
| CNN2 con augmentation | 84.90% | 84.86% |
| Random Forest con HOG + color | 65.94% | 64.96% |
| CNN2 sin augmentation | 61.34% | 58.64% |

La prueba externa contiene **150 fotos propias y 15 letras**. Cada uno de los tres
integrantes aportó 50 imágenes de cinco letras distintas. La CNN con augmentation
alcanzó 36% global; la CNN sin augmentation obtuvo 6% y Random Forest 0%. Esta
caída fuera del dataset evidencia cambio de dominio y respalda la necesidad de
datos más diversos.

## Cómo reproducir

Ejecute desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Lab3\requirements.txt
.\.venv\Scripts\python.exe Lab3\src\prepare_asl_data.py
.\.venv\Scripts\python.exe Lab3\src\random_forest_pipeline.py
```

La descarga estándar de Kaggle debe estar extraída como alguna de estas rutas:

```text
Lab3/data/asl_alphabet_train/<clases>/
Lab3/data/asl_alphabet_train/asl_alphabet_train/<clases>/
```

Los datos crudos, arreglos `.npy`, características cacheadas y el ZIP se excluyen
de Git por su tamaño. Los resultados, scripts y modelos finales sí se versionan.

## Fotos propias del grupo

El pipeline detecta fotografías dentro de `Lab3/asl_alphabet_test/<letra>/`.

| Integrante | Letras | Fotos |
|---|---|---:|
| Ricardo  | I, J, K, L, R | 50 |
| Sebastian | M, N, S, U, X | 50 |
| Vianka  | O, V, W, Y, Z | 50 |

