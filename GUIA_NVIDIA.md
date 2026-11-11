# Guía oficial para ejecutar ASL-CUDA-Optimization en NVIDIA

Esta es la secuencia canónica para Windows 11. CPU y CUDA se miden en la **misma computadora NVIDIA**, con los mismos datos y checkpoint. Los scripts no descargan el dataset y los benchmarks completos requieren confirmación explícita.

## 1. Clonar y comprobar Python/GPU

Abra PowerShell y ejecute:

```powershell
git clone https://github.com/Vann06/ASL-CUDA-Optimization.git
Set-Location ASL-CUDA-Optimization
git status --short
py -3.11 --version
nvidia-smi
```

Se requiere Python **3.11.9 x64**. Si falta, instálelo desde Python.org, abra otra terminal y repita. Si `nvidia-smi` falla, repare o actualice el driver NVIDIA antes de continuar. No hace falta instalar CUDA Toolkit: la rueda PyTorch incluye el runtime CUDA 12.6.

## 2. Crear los dos entornos

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_cpu.ps1
.\scripts\setup_cuda.ps1
```

Los scripts crean `.venv-cpu` y `.venv-cuda`, exigen Python 3.11.9, instalan las versiones exactas, ejecutan `pip check`, prueban el dispositivo y guardan `pip freeze`/hardware en `results/environment/`. No mezcle los dos requirements en un entorno.

## 3. Obtener ASL Alphabet y regenerar los datos

Los datos no están en Git. Obtenga legítimamente **ASL Alphabet** de Grassknoted en Kaggle: <https://www.kaggle.com/datasets/grassknoted/asl-alphabet>. Descargue y extraiga manualmente fuera del repositorio. La ruta elegida debe contener una carpeta `asl_alphabet_train` con 29 subcarpetas y 3,000 imágenes por clase.

Asigne su ruta local a una variable; no edite código:

```powershell
$DatasetRoot = Read-Host "Ruta de la carpeta extraída de ASL Alphabet"
.\.venv-cpu\Scripts\python.exe original_lab3/src/prepare_asl_data.py --source-root $DatasetRoot --output-dir data
.\.venv-cpu\Scripts\python.exe scripts/verify_data.py --data-dir data --reference evidence/asl_data_reference.json --output results/environment/data_manifest.json
```

El preparador valida 87,000 archivos, ordena las 29 clases, normaliza `delete` a `del`, usa seed 42, selecciona 20 bloques de 30 por clase y produce:

| Split | Imágenes | Forma | Tipo |
|---|---:|---|---|
| train | 12,180 | `(12180,64,64,3)` | `uint8` |
| val | 2,610 | `(2610,64,64,3)` | `uint8` |
| test | 2,610 | `(2610,64,64,3)` | `uint8` |

El orden esperado empieza `a,b,c,d,del,e` y termina `w,x,y,z`; el listado completo y los SHA-256 están en `evidence/asl_data_reference.json`. La verificación falla ante cualquier diferencia de clase, dimensión, tipo, tamaño o hash. No continúe si falla: confirme que usó la versión estándar de 3,000 imágenes por clase, Pillow 11.1.0 y una extracción íntegra. Conserve el manifiesto divergente para diagnóstico; no sustituya la referencia ni asuma equivalencia.

## 4. Verificar checkpoint y archivos ignorados

```powershell
git ls-files -- results/initialization/cnn2_seed42_initial.pt
Get-FileHash results/initialization/cnn2_seed42_initial.pt -Algorithm SHA256
git status --short
```

El primer comando debe listar el archivo. Hash esperado:

```text
B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F
```

Los entornos, `data/`, `.npy`, resultados generados y evidencias locales de hardware están ignorados. `git status` no debe mostrar esos archivos.

## 5. Ejecutar smoke tests

```powershell
.\scripts\run_smoke_tests.ps1 -Target both
```

Esto verifica hashes, datos, 11 pruebas CPU, augmentation, `pip check`, CUDA real y una ejecución reducida por dispositivo. No representa resultados del proyecto. Cada salida se guarda separada en `results/benchmarks_smoke/cpu/` o `cuda/` y se valida automáticamente.

No continúe hasta completar las secciones A y B de [`CHECKLIST_NVIDIA.md`](CHECKLIST_NVIDIA.md).

## 6. Preparar las capturas

No ejecute CPU y CUDA simultáneamente. Use el mismo plan de energía, conecte el equipo a corriente y cierre aplicaciones pesadas. Abra Administrador de tareas → Rendimiento y ajuste sus ventanas para que no muestren usuario, correo, tokens ni rutas privadas.

Tome estas capturas:

1. `01_nvidia_smi.png`: `nvidia-smi` antes de medir; debe verse GPU y driver.
2. `02_smoke_cpu_cuda.png`: final exitoso de ambos smoke tests.
3. `03_cpu_entrenamiento.png`: Administrador de tareas durante CPU; CPU y memoria RAM visibles.
4. `04_ram_cpu.png`: detalle del máximo aproximado de RAM durante CPU.
5. `05_gpu_entrenamiento.png`: Administrador de tareas durante CUDA, gráfica CUDA/Compute visible.
6. `06_nvidia_smi_cuda.png`: `nvidia-smi -l 1` en otra terminal durante CUDA; utilización y memoria GPU visibles.
7. `07_memoria_gpu.png`: evidencia clara de VRAM usada/pico observado.
8. `08_benchmark_final.png`: final exitoso y rutas de resultados.

Guárdelas fuera de Git en `results/evidence_local/`. Para observar GPU durante el benchmark, abra antes otra terminal y ejecute `nvidia-smi -l 1`; deténgalo con Ctrl+C al finalizar CUDA.

## 7. Ejecutar el protocolo oficial

El siguiente comando tarda y entrena 10 épocas × 3 repeticiones en CPU y luego CUDA. Sin el interruptor de confirmación no hace nada.

```powershell
.\scripts\run_full_benchmarks.ps1 -ConfirmFullBenchmark
```

Parámetros fijados: seed 42, batch 32, `num_workers=0`, dos batches de warm-up excluidos, sin EarlyStopping y sin límites de batches. El mismo checkpoint se restaura tras warm-up y antes de cada repetición.

Cada directorio contiene:

- `benchmark.json`: estado/errores, dispositivo, versiones, hardware, configuración, SHA del checkpoint, definiciones y tiempos completos;
- `epochs.csv`: dispositivo, métricas y tiempos por fase/época;
- `runs.csv`: dispositivo, parámetros, accuracy/F1 y tiempos por repetición;
- `benchmark.log`: progreso o traza de error;
- `error.json`: solo si una ejecución iniciada falla.

Los tiempos internos son exclusivos y no deben sumarse al tiempo pared: `epoch_wall_s` ya contiene carga, transferencia, augmentation, forward, backward y optimizer. `training_wall_s` contiene épocas y validaciones; `measured_execution_s` añade restauración y prueba final, excluyendo warm-up; `run_wall_s` incluye setup y warm-up. `overall_process_wall_s` cubre las tres repeticiones.

## 8. Validar y recuperar resultados

El script completo valida automáticamente la última salida. Para repetir la validación manual, sustituya las rutas por los directorios UTC creados:

```powershell
.\.venv-cpu\Scripts\python.exe scripts/validate_results.py results/benchmarks/cpu/AAAAmmddTHHMMSSZ_cpu --expect-device cpu --expect-full
.\.venv-cuda\Scripts\python.exe scripts/validate_results.py results/benchmarks/cuda/AAAAmmddTHHMMSSZ_cuda --expect-device cuda --expect-full
Get-ChildItem results/benchmarks -Recurse
```

Copie antes de borrar entornos:

- `results/benchmarks/`;
- `results/benchmarks_smoke/`;
- `results/environment/`;
- `results/evidence_local/`;
- [`CHECKLIST_NVIDIA.md`](CHECKLIST_NVIDIA.md) marcada en una copia local.

Comprima sin incluir dataset ni entornos:

```powershell
Compress-Archive -Path results/benchmarks,results/benchmarks_smoke,results/environment,results/evidence_local -DestinationPath benchmark_entrega.zip
```

El ZIP y resultados están ignorados por Git. Entréguelos por el canal acordado.

## 9. Problemas frecuentes

- **Python distinto de 3.11.9:** instale la versión exacta y elimine manualmente solo el entorno virtual equivocado; los scripts nunca borran entornos.
- **PowerShell bloquea scripts:** use `Set-ExecutionPolicy -Scope Process Bypass`; no cambie la política del sistema.
- **CUDA no disponible:** compruebe `nvidia-smi`, `.venv-cuda`, `torch==2.6.0+cu126` y driver. No use CPU como sustituto.
- **Hash de datos distinto:** no edite el manifiesto. Revise dataset, extracción, nombres, conteos y Pillow; entregue el manifiesto generado y el error.
- **Benchmark interrumpido:** conserve `benchmark.log` y `error.json`. No combine una repetición parcial con otra ejecución.
- **Falta espacio:** no mueva `data/` al repositorio ni lo suba a Git; libere espacio antes de ejecutar.
