# Discrepancia del entorno activo — 2026-11-11

## Objetivo fijado

- Python 3.11.9
- PyTorch 2.6.0 (`+cpu` o `+cu126`)
- torchvision 0.21.0
- NumPy 1.26.4

## Estado observado

1. No había entorno virtual activo: `VIRTUAL_ENV` estaba vacío.
2. `python` resolvía a `C:\Users\viank\AppData\Local\Programs\Python\Python310\python.exe`.
3. El intérprete activo era Python 3.10.11.
4. `.venv` también fue creado con Python 3.10.11 y no contiene PyTorch.
5. El site-packages global de Python 3.10 contiene una distribución inválida `-orch`; `pip show torch` no encuentra el paquete y `import torch` falla dentro de `torch.fx.passes`.
6. La ejecución anterior que reportó PyTorch 2.14.0 ya no es reproducible con ese intérprete global.
7. Existe Python 3.13.7 con PyTorch 2.13.0+cpu y NumPy 2.4.1, sin torchvision. Se utilizó únicamente para smoke tests no canónicos.

## Impacto

El benchmark oficial queda bloqueado hasta activar un entorno limpio Python 3.11.9 con los requisitos fijados. No se cambiaron dependencias ni se reparó el Python global, para evitar mezclar instalaciones.

## Validación posible en el intérprete alterno

- El checkpoint se cargó con `weights_only=True`.
- Todos los tensores cargados coincidieron exactamente con el `state_dict` guardado.
- Una nueva inicialización con semilla 42 coincidió con el checkpoint.
- SHA-256: `B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F`.

Estos resultados no sustituyen la repetición obligatoria bajo Python 3.11.9/PyTorch 2.6.0.
