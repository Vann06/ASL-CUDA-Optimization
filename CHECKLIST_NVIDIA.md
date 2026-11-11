# Lista de verificación NVIDIA

Marque cada casilla solo con evidencia real de la misma computadora NVIDIA.

## A. Entorno listo

- [ ] Clon limpio en Windows 11 x64; `git status --short` no muestra cambios de código.
- [ ] Python 3.11.9 x64 disponible con `py -3.11`.
- [ ] `setup_cpu.ps1` terminó sin errores y `pip check` pasó.
- [ ] `setup_cuda.ps1` terminó sin errores; `nvidia-smi` y CUDA real pasaron.
- [ ] Checkpoint SHA-256: `B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F`.
- [ ] Los seis `.npy`, `classes.json` y `preprocess_meta.json` coinciden con el manifiesto versionado.
- [ ] Se guardaron `pip-freeze-cpu.txt`, `pip-freeze-cuda.txt` y reportes de hardware.

## B. Smoke tests aprobados

- [ ] `run_smoke_tests.ps1 -Target both` terminó con código 0.
- [ ] 11 pruebas unitarias CPU pasaron.
- [ ] Smoke CPU indica `full_protocol: false` y `status: completed`.
- [ ] Smoke CUDA indica `full_protocol: false`, CUDA disponible y `status: completed`.
- [ ] `validate_results.py` aprobó ambos directorios.
- [ ] La preview confirma augmentation solo en entrenamiento; validación/prueba permanecen deterministas.

## C. Benchmarks oficiales completos

- [ ] Se cerraron aplicaciones pesadas y se mantuvo el mismo plan de energía.
- [ ] Se ejecutó una sola vez `run_full_benchmarks.ps1 -ConfirmFullBenchmark`.
- [ ] CPU y CUDA tienen 10 épocas, batch 32, tres repeticiones, seed 42 y `num_workers=0`.
- [ ] Ambos `benchmark.json` indican `full_protocol: true`, `status: completed` y `errors: []`.
- [ ] Existen `epochs.csv`, `runs.csv`, `benchmark.json` y `benchmark.log` para ambos dispositivos.
- [ ] Se tomaron las capturas solicitadas y se copiaron resultados antes de limpiar entornos.
- [ ] No se incluyeron datos, `.npy`, entornos virtuales, nombres de usuario ni tokens en la entrega.
