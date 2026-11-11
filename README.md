# Proyecto 3 — Optimización de reconocimiento ASL con CUDA

Este repositorio parte del **Laboratorio 3 de reconocimiento del alfabeto ASL**, implementado con TensorFlow/Keras. El objetivo del Proyecto 3 es migrar la **CNN2 con data augmentation** a PyTorch y estudiar su ejecución en CPU y GPU/CUDA sin perder trazabilidad respecto del experimento original.

## Estado actual

La auditoría reproducible del Lab3 está completa, existe una configuración fijada para Windows 11 y la arquitectura CNN2 ya está traducida a PyTorch. Todavía no se entrena el dataset completo ni se publican benchmarks CPU/CUDA.

- Auditoría técnica: [`notebooks/00_auditoria_lab3.ipynb`](notebooks/00_auditoria_lab3.ipynb)
- Laboratorio original: [`original_lab3/`](original_lab3/)
- Notebook original de modelos: [`original_lab3/Notebooks/01_modelos_dl.ipynb`](original_lab3/Notebooks/01_modelos_dl.ipynb)
- Modelo objetivo guardado: `original_lab3/Notebooks/mejor_modelo_aug.keras`
- Guía de entorno CPU/GPU: [`GUIA_EJECUCION.md`](GUIA_EJECUCION.md)
- Verificador y smoke test: [`scripts/check_environment.py`](scripts/check_environment.py)
- Implementación CNN2: [`src/models/cnn2.py`](src/models/cnn2.py)
- Dataset y augmentation: [`src/data/asl_dataset.py`](src/data/asl_dataset.py)
- Notebook PyTorch: [`notebooks/01_modelo_pytorch.ipynb`](notebooks/01_modelo_pytorch.ipynb)
- Pruebas CPU: [`tests/test_model.py`](tests/test_model.py)

## CNN2 en PyTorch

La traducción conserva tres bloques Conv2D con ReLU–BatchNorm–MaxPool–Dropout, Global Average Pooling, dropout final y clasificación de 29 clases. PyTorch registra **97,437 parámetros entrenables**; al sumar las 448 medias y varianzas móviles de BatchNorm que Keras cuenta como no entrenables, el total comparable es **97,885**.

Los `.npy` se leen con memoria mapeada. Cada imagen NHWC `uint8` se convierte bajo demanda a NCHW `float32` en `[0,1]`, evitando una copia float32 completa de aproximadamente 570.94 MiB para entrenamiento.

Validación rápida, sin entrenar épocas completas:

```powershell
python -m unittest -v tests.test_model
python scripts/create_initial_state.py
```

El checkpoint `results/initialization/cnn2_seed42_initial.pt` contiene el estado inicial con semilla 42 que deberán compartir los benchmarks CPU y CUDA.

## Entorno reproducible

El proyecto usa Python 3.11.9, NumPy 1.26.4, PyTorch 2.6.0 y torchvision 0.21.0. Hay dos instalaciones excluyentes:

```powershell
# Computadora sin NVIDIA
python -m pip install -r requirements-cpu.txt
python scripts/check_environment.py --device cpu

# Computadora con NVIDIA y driver compatible
python -m pip install -r requirements-cuda.txt
python scripts/check_environment.py --device cuda
```

Las ruedas CPU y CUDA 12.6 para CPython 3.11/Windows x64 fueron verificadas en los índices oficiales. Consulte la guía antes de instalar; no mezcle ambos archivos en un mismo entorno.

## Alcance de la auditoría

La auditoría documenta:

1. carga, selección por bloques, partición, preprocesamiento y normalización;
2. augmentation, entrenamiento y evaluación;
3. arquitectura y parámetros exactos de `CNN2_aug`;
4. operaciones repetitivas y fórmulas dependientes de épocas y batch size;
5. responsabilidades recomendadas para CPU y GPU/CUDA;
6. artefactos disponibles y faltantes;
7. diferencias que deben resolverse al traducir de Keras a PyTorch.

## Reproducibilidad

El Lab3 original se conserva como referencia y no debe modificarse. Las rutas del notebook de auditoría son relativas a la raíz del repositorio y sus verificaciones solo leen metadatos, notebooks, métricas y modelos existentes. El dataset de entrenamiento y los arreglos `.npy` no están versionados, por lo que actualmente no es posible reconstruir la partición ni repetir el entrenamiento desde cero.

Los registros locales de `pip freeze` y hardware deben guardarse en `results/environment/`. Esa carpeta está ignorada por Git para evitar publicar detalles locales sensibles.

## Próximas etapas

La siguiente etapa es validar el mismo checkpoint en el entorno fijado PyTorch 2.6.0 y en la computadora NVIDIA, y después ejecutar benchmarks controlados en notebooks separados. No se deben reutilizar las métricas históricas como si fueran mediciones de la implementación PyTorch.
