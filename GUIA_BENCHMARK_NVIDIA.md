# Benchmark CPU y CUDA en la misma computadora NVIDIA

Esta guía se conserva como enlace compatible. La secuencia canónica y actualizada, desde el clon hasta la recuperación de evidencias, está en [`GUIA_NVIDIA.md`](GUIA_NVIDIA.md).

Comandos oficiales, desde la raíz del repositorio:

```powershell
.\scripts\setup_cpu.ps1
.\scripts\setup_cuda.ps1
.\scripts\run_smoke_tests.ps1 -Target both
.\scripts\run_full_benchmarks.ps1 -ConfirmFullBenchmark
```

El último comando ejecuta 10 épocas, batch 32 y tres repeticiones primero en CPU y después en CUDA. Sin `-ConfirmFullBenchmark` termina antes de entrenar.
