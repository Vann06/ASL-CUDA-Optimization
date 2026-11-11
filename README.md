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
- Benchmark modular: [`src/train.py`](src/train.py) y [`src/benchmark.py`](src/benchmark.py)
- Notebooks de benchmark: [`notebooks/02_benchmark_cpu.ipynb`](notebooks/02_benchmark_cpu.ipynb) y [`notebooks/03_benchmark_cuda.ipynb`](notebooks/03_benchmark_cuda.ipynb)
- Guía para medir CPU/CUDA en la misma NVIDIA: [`GUIA_BENCHMARK_NVIDIA.md`](GUIA_BENCHMARK_NVIDIA.md)
- Guía canónica de entrega para el compañero NVIDIA: [`GUIA_NVIDIA.md`](GUIA_NVIDIA.md)
- Lista de verificación final: [`CHECKLIST_NVIDIA.md`](CHECKLIST_NVIDIA.md)

## CNN2 en PyTorch

La traducción conserva tres bloques Conv2D con ReLU–BatchNorm–MaxPool–Dropout, Global Average Pooling, dropout final y clasificación de 29 clases. PyTorch registra **97,437 parámetros entrenables**; al sumar las 448 medias y varianzas móviles de BatchNorm que Keras cuenta como no entrenables, el total comparable es **97,885**.

Los `.npy` se leen con memoria mapeada. Cada imagen NHWC `uint8` se convierte bajo demanda a NCHW `float32` en `[0,1]`, evitando una copia float32 completa de aproximadamente 570.94 MiB para entrenamiento.

Validación rápida, sin entrenar épocas completas:

```powershell
python -m unittest -v tests.test_model
python scripts/create_initial_state.py
```

El checkpoint `results/initialization/cnn2_seed42_initial.pt` contiene el estado inicial con semilla 42 que deberán compartir los benchmarks CPU y CUDA.
Está versionado y su SHA-256 obligatorio es `B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F`.

## Benchmark reproducible

El protocolo oficial queda preparado, pero todavía no se ha ejecutado completo:

- 10 épocas fijas, sin EarlyStopping;
- batch size 32;
- tres repeticiones;
- mismo checkpoint y seed 42 en cada repetición;
- warm-up excluido y pesos restaurados después;
- tiempos exclusivos de carga, transferencia, augmentation, forward, backward y optimizer;
- CSV/JSON con métricas, hardware, versiones y configuración.

Smoke test CPU limitado:

```powershell
python -m src.benchmark --device cpu --epochs 1 --batch-size 32 --repetitions 1 --num-workers 0 --seed 42 --warmup-batches 1 --max-train-batches 2 --max-eval-batches 1 --output-dir results/benchmarks_smoke/cpu
```

En la computadora NVIDIA, los scripts oficiales evitan diferencias manuales:

```powershell
.\scripts\setup_cpu.ps1
.\scripts\setup_cuda.ps1
.\scripts\run_smoke_tests.ps1 -Target both
.\scripts\run_full_benchmarks.ps1 -ConfirmFullBenchmark
```

Use `--device cuda` únicamente con la rueda CUDA y una GPU NVIDIA disponible. La solicitud falla antes de crear resultados si CUDA no está disponible.

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

El Lab3 original se conserva como referencia y no debe modificarse. El dataset y los arreglos `.npy` no están versionados. Desde un clon limpio se obtiene legalmente **ASL Alphabet**, se extrae fuera de Git y se ejecuta `original_lab3/src/prepare_asl_data.py`; el script usa seed 42, orden de clases estable y splits por bloques. `scripts/verify_data.py` compara formas, tipos, orden y SHA-256 contra [`evidence/asl_data_reference.json`](evidence/asl_data_reference.json) y falla si existe cualquier diferencia.

Los registros locales de `pip freeze` y hardware deben guardarse en `results/environment/`. Esa carpeta está ignorada por Git para evitar publicar detalles locales sensibles.

## Próximas etapas

La siguiente etapa se ejecuta únicamente en la computadora NVIDIA siguiendo [`GUIA_NVIDIA.md`](GUIA_NVIDIA.md). No se deben reutilizar los smoke tests no canónicos ni las métricas históricas como resultados oficiales.
