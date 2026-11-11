# Proyecto 3 — Optimización de reconocimiento ASL con CUDA

Este repositorio parte del **Laboratorio 3 de reconocimiento del alfabeto ASL**, implementado con TensorFlow/Keras. El objetivo del Proyecto 3 es migrar la **CNN2 con data augmentation** a PyTorch y estudiar su ejecución en CPU y GPU/CUDA sin perder trazabilidad respecto del experimento original.

## Estado actual

La primera etapa es una auditoría reproducible del Lab3. Todavía no se implementa PyTorch/CUDA ni se reentrenan modelos.

- Auditoría técnica: [`notebooks/00_auditoria_lab3.ipynb`](notebooks/00_auditoria_lab3.ipynb)
- Laboratorio original: [`original_lab3/`](original_lab3/)
- Notebook original de modelos: [`original_lab3/Notebooks/01_modelos_dl.ipynb`](original_lab3/Notebooks/01_modelos_dl.ipynb)
- Modelo objetivo guardado: `original_lab3/Notebooks/mejor_modelo_aug.keras`

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

## Próximas etapas

Después de cerrar las decisiones de paridad Keras→PyTorch se implementarán, en notebooks separados, el modelo PyTorch, los benchmarks CPU/CUDA y el análisis de resultados. No se deben reutilizar las métricas históricas como si fueran mediciones de la futura implementación.
