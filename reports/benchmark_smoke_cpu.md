# Smoke test CPU del sistema de benchmarks

Fecha: 2026-11-11  
Estado: **prueba rápida aprobada; no es benchmark oficial**.

## Entorno usado

- Python 3.13.7
- PyTorch 2.13.0+cpu
- NumPy 2.4.1
- scikit-learn 1.8.0
- CUDA no disponible

Este entorno no coincide con los pines Python 3.11.9/PyTorch 2.6.0. Se utilizó solo porque el Python 3.10 activo tiene una instalación `torch` inválida. El protocolo completo sigue bloqueado.

## Pruebas

```powershell
python -m unittest -v tests.test_model tests.test_benchmark
```

Resultado: 11/11 pruebas aprobadas en 9.363 s. Incluye un batch real y una época reducida a dos batches train y un batch de validación/test.

## Ejecución CLI limitada

Configuración: una época, batch 32, una repetición, un batch de warm-up, dos batches train y un batch de evaluación. El JSON marca `full_protocol: false`.

Directorio:

```text
results/benchmarks_smoke/20261111T164547Z_cpu/
```

Tiempos observados:

| Medición | Segundos |
|---|---:|
| Setup de datos/loaders | 0.0221 |
| Warm-up excluido | 6.2107 |
| Restauración del modelo | 0.0373 |
| Época reducida + validación | 0.8565 |
| Evaluación final limitada | 0.1202 |
| Ejecución medida | 1.0143 |
| Repetición completa | 7.2593 |
| Overhead no atribuido entre agregados | 0.0122 |

La accuracy y el F1 del subset no se interpretan: parten de pesos iniciales y la prueba solo procesa una fracción sesgada del dataset. No se generó ningún resultado CUDA.
