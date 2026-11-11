# Guía de ejecución reproducible en Windows 11

Esta guía prepara el mismo proyecto para integrantes con solo CPU y para una computadora con GPU NVIDIA. No instala TensorFlow, no descarga datasets y no entrena modelos.

Para la medición oficial en la computadora NVIDIA, la fuente canónica es [`GUIA_NVIDIA.md`](GUIA_NVIDIA.md). Los scripts `setup_cpu.ps1` y `setup_cuda.ps1` crean entornos separados; no use un único `.venv311` para ambas variantes.

## Versiones fijadas

| Componente | Versión |
|---|---:|
| Python | 3.11.9 |
| PyTorch | 2.6.0 |
| torchvision | 0.21.0 |
| CUDA de la rueda GPU | 12.6 |
| NumPy | 1.26.4 |

Python 3.11.9 se eligió porque es la última versión 3.11 con instalador binario oficial para Windows. Las ruedas oficiales verificadas para CPython 3.11 y Windows x64 son:

- `torch-2.6.0+cpu-cp311-cp311-win_amd64.whl`
- `torchvision-0.21.0+cpu-cp311-cp311-win_amd64.whl`
- `torch-2.6.0+cu126-cp311-cp311-win_amd64.whl`
- `torchvision-0.21.0+cu126-cp311-cp311-win_amd64.whl`

PyTorch publica la combinación 2.6.0/0.21.0 para CPU y CUDA 12.6 en su [tabla oficial de versiones anteriores](https://pytorch.org/get-started/previous-versions/). Los índices usados son `https://download.pytorch.org/whl/cpu` y `https://download.pytorch.org/whl/cu126`.

## 1. Requisitos previos

1. Windows 11 x64.
2. Git.
3. Python 3.11.9 x64 desde el [instalador oficial](https://www.python.org/downloads/release/python-3119/), o `pyenv-win` respetando `.python-version`.
4. Solo para GPU: una GPU NVIDIA y un driver NVIDIA reciente compatible con CUDA 12.6. Para las ruedas de PyTorch no se necesita instalar el CUDA Toolkit completo.

Compruebe el intérprete antes de crear el entorno:

```powershell
py -3.11 --version
py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
```

La salida esperada es Python 3.11.x y `64`.

> No reutilice un `.venv` creado con Python 3.10. Un entorno virtual no cambia de versión de Python al actualizar paquetes.

## 2. Crear entornos limpios

Desde la raíz de `ASL-CUDA-Optimization`:

```powershell
.\scripts\setup_cpu.ps1
# Solo en una computadora NVIDIA:
.\scripts\setup_cuda.ps1
```

Si PowerShell bloquea la activación, se puede ejecutar el Python del entorno directamente:

```powershell
.\.venv-cpu\Scripts\python.exe --version
.\.venv-cuda\Scripts\python.exe --version
```

## 3A. Instalación para CPU

Use esta opción en computadoras sin NVIDIA. No instale la variante CUDA para “tenerla disponible”; ocupa más espacio y no convierte una computadora sin NVIDIA en una computadora CUDA.

```powershell
python -m pip install -r requirements-cpu.txt
python -m pip check
python scripts/check_environment.py --device cpu --output results/environment/hardware-cpu.json
```

El reporte correcto debe mostrar:

- `torch_version`: `2.6.0+cpu`;
- `torchvision_version`: `0.21.0+cpu`;
- `cuda_compiled`: `null`;
- `cuda_available`: `false`;
- smoke test en `cpu` con estado `passed`.

## 3B. Instalación para GPU NVIDIA

Primero verifique que Windows detecta la GPU y el driver:

```powershell
nvidia-smi
```

Después instale la rueda CUDA 12.6:

```powershell
python -m pip install -r requirements-cuda.txt
python -m pip check
python scripts/check_environment.py --device cuda --output results/environment/hardware-cuda.json
```

El reporte correcto debe mostrar:

- `torch_version`: `2.6.0+cu126`;
- `torchvision_version`: `0.21.0+cu126`;
- `cuda_compiled`: `12.6`;
- `cuda_available`: `true`;
- nombre de GPU y versión del driver;
- smoke test en `cuda` con estado `passed`.

`--device cuda` es estricto: si CUDA no está disponible, el script termina con código distinto de cero y no ejecuta la prueba silenciosamente en CPU. `--device auto` es útil para diagnóstico, pero los registros oficiales del proyecto deben usar `cpu` o `cuda` explícitamente.

## 4. Registrar dependencias y hardware

Después de que `pip check` y el smoke test pasen:

```powershell
New-Item -ItemType Directory -Force results/environment | Out-Null
python -m pip freeze --all | Out-File -Encoding utf8 results/environment/pip-freeze.txt
python scripts/check_environment.py --device cpu --output results/environment/hardware.json
```

En la computadora NVIDIA, cambie `--device cpu` por `--device cuda`. Conserve estos archivos localmente o entréguelos por el canal acordado para evidencia. `results/environment/` está ignorado por Git porque puede revelar detalles de hardware y software local.

Para comparar tres integrantes, cada persona debe registrar como mínimo:

- `pip-freeze.txt`;
- `hardware.json`;
- salida exitosa de `python -m pip check`.

## 5. Uso en JupyterLab

Con el entorno activo:

```powershell
python -m ipykernel install --user --name asl-cuda-py311 --display-name "ASL CUDA (Python 3.11)"
python -m jupyter lab
```

Seleccione siempre el kernel **ASL CUDA (Python 3.11)**. El registro `--user` es local y no se versiona.

## 6. Problemas frecuentes

### `py -3.11` no existe

Instale Python 3.11.9 x64 y abra una terminal nueva. Compruebe la ruta con:

```powershell
Get-Command py
py -0p
```

### El entorno sigue usando Python 3.10

Elimine solo el entorno virtual equivocado después de confirmar su ruta y vuelva a ejecutar `setup_cpu.ps1` o `setup_cuda.ps1`. No modifique el Python del sistema ni `original_lab3/`.

### `torch.cuda.is_available()` devuelve `False`

Compruebe, en este orden:

1. que se instaló `requirements-cuda.txt`, no `requirements-cpu.txt`;
2. que `python -c "import torch; print(torch.__version__, torch.version.cuda)"` muestre `2.6.0+cu126 12.6`;
3. que `nvidia-smi` detecte la GPU y muestre el driver;
4. que el comando use `.venv-cuda\Scripts\python.exe`;
5. que el driver NVIDIA esté actualizado.

No instale paquetes llamados `cuda`, `cudatoolkit` o `pytorch-cuda` con pip para corregir esta instalación: las ruedas oficiales indicadas ya incluyen el runtime que necesitan.

### Se instaló la variante equivocada de PyTorch

Es más seguro recrear únicamente el entorno equivocado. No mezcle las ruedas CPU y CUDA.

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install -r requirements-cpu.txt
```

Use `requirements-cuda.txt` en lugar de `requirements-cpu.txt` solo en la computadora NVIDIA.

### Errores de DLL al importar PyTorch

Confirme Windows x64, Python x64 y todas las actualizaciones de Windows. Instale o repare Microsoft Visual C++ Redistributable x64. Después recree el entorno; no copie DLL manualmente dentro del repositorio.

### `pip check` reporta conflictos

No continúe con benchmarks. Guarde la salida, recree un entorno vacío y reinstale un único archivo de requisitos. No mezcle CPU y CUDA en el mismo entorno.

## 7. Validación de ruedas realizada

Se consultaron los índices oficiales con compatibilidad objetivo `--platform win_amd64 --python-version 3.11 --implementation cp --abi cp311`. Se verificaron los cuatro artefactos PyTorch indicados arriba y las versiones directas fijadas en `requirements.txt`. Esta validación demuestra disponibilidad de distribución; la prueba final de driver y ejecución CUDA queda pendiente en la computadora NVIDIA.

## 8. Validar CNN2 y los datos

Estas comprobaciones usan un solo batch de 32 imágenes; no entrenan épocas completas:

```powershell
python -m unittest -v tests.test_model
python scripts/create_initial_state.py
```

Las pruebas cubren:

- formas NCHW, etiquetas y rango `[0,1]`;
- augmentation de entrenamiento y ausencia de cambios aleatorios en evaluación;
- salida `(32,29)` y valores finitos;
- 97,437 parámetros entrenables y 97,885 bajo el conteo compatible con Keras;
- forward, backward y una actualización Adam;
- carga exacta del checkpoint inicial;
- error obligatorio si se pide CUDA y no está disponible.

El estado inicial canónico queda en:

```text
results/initialization/cnn2_seed42_initial.pt
```

Los benchmarks deben cargar ese archivo antes de mover el modelo al dispositivo. No deben crear pesos independientes para CPU y GPU.

Verifique que todos tengan el mismo archivo:

```powershell
Get-FileHash results/initialization/cnn2_seed42_initial.pt -Algorithm SHA256
```

SHA-256 esperado: `B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F`.

### Comprobación CPU

```powershell
python scripts/check_environment.py --device cpu --output results/environment/hardware-cpu.json
python -m unittest -v tests.test_model
```

### Comprobación pendiente en NVIDIA

```powershell
python scripts/check_environment.py --device cuda --output results/environment/hardware-cuda.json
python -c "from src.utils.device import select_device; print(select_device('cuda'))"
python -m unittest -v tests.test_model
```

Las pruebas unitarias actuales fuerzan el paso de optimización en CPU para mantener una referencia común. El benchmark CUDA se implementará aparte y deberá cargar el mismo checkpoint.

## 9. Diferencias declaradas respecto de TensorFlow/Keras

- PyTorch usa NCHW; el Lab3 usaba NHWC.
- `CrossEntropyLoss` recibe logits, por lo que el modelo no aplica softmax durante entrenamiento.
- BatchNorm usa `eps=1e-3` y `momentum=0.01`, equivalente a la convención Keras `momentum=0.99`.
- TensorFlow y PyTorch no comparten RNG ni implementación exacta de interpolación; se conservan rangos, orden, bilinear y reflection padding, pero no se espera igualdad píxel a píxel en augmentation.
- Se corrigió deliberadamente el contraste para datos normalizados: factor `[0.8,1.2]` y clipping `[0,1]`. El Lab3 dejó el rango predeterminado `[0,255]` en `RandomContrast`.
- PyTorch cuenta 97,437 parámetros porque las estadísticas móviles de BatchNorm son buffers. Sumadas como hace Keras, el total es 97,885.

## 10. Benchmarks

El sistema de medición está documentado en [`GUIA_BENCHMARK_NVIDIA.md`](GUIA_BENCHMARK_NVIDIA.md). La comparación oficial debe ejecutarse en la misma computadora NVIDIA usando entornos CPU y CUDA separados y el mismo checkpoint.

Antes de cualquier ejecución completa:

```powershell
python scripts/verify_data.py
python -m unittest -v tests.test_model tests.test_benchmark
python scripts/preview_augmentation.py --device cpu
```

No ejecute las diez épocas si el entorno, los hashes, el smoke test o `pip check` no pasan.
