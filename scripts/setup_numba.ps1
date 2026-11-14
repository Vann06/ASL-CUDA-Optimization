[CmdletBinding()]
param()

# Crea .venv-numba para los kernels CUDA escritos a mano (src/cuda_kernels).
# Las ruedas pip de NVIDIA instalan NVVM y cudart en carpetas separadas; Numba espera la
# estructura de un toolkit CUDA, así que se arma .venv-numba\cuda_home con ambas partes.
# src/cuda_kernels/photometric.py apunta CUDA_HOME a esa carpeta automáticamente.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExpected = "3.11.9"
$Venv = Join-Path $RepoRoot ".venv-numba"

Push-Location $RepoRoot
try {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi no está disponible. Instale/verifique el driver NVIDIA."
    }
    if (-not (Test-Path $Venv)) {
        & py -3.11 -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "No se pudo crear .venv-numba." }
    }
    $Python = Join-Path $Venv "Scripts\python.exe"
    $VenvVersion = & $Python -c "import platform; print(platform.python_version())"
    if ($VenvVersion.Trim() -ne $PythonExpected) {
        throw ".venv-numba usa Python $VenvVersion; se requiere $PythonExpected."
    }
    & $Python -m pip install --upgrade pip==25.0.1
    if ($LASTEXITCODE -ne 0) { throw "Falló la actualización fijada de pip." }
    & $Python -m pip install -r requirements-numba.txt
    if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de requirements-numba.txt." }

    $Nvidia = Join-Path $Venv "Lib\site-packages\nvidia"
    $CudaHome = Join-Path $Venv "cuda_home"
    New-Item -ItemType Directory -Force -Path (Join-Path $CudaHome "bin") | Out-Null
    Copy-Item (Join-Path $Nvidia "cuda_runtime\bin\cudart64_12.dll") (Join-Path $CudaHome "bin") -Force
    Copy-Item (Join-Path $Nvidia "cuda_nvcc\nvvm") $CudaHome -Recurse -Force

    $Check = "from src.cuda_kernels import photometric as P; from numba import cuda; " +
             "import numpy as np; x=np.random.rand(4,3,64,64).astype(np.float32); " +
             "d,f=P.sample_parameters(4,np.random.default_rng(0)); " +
             "e=float(np.abs(P.photometric_cuda(x,d,f)-P.photometric_numpy(x,d,f)).max()); " +
             "print(cuda.get_current_device().name, 'diff', e); assert e < 1e-5"
    & $Python -W ignore -c $Check
    if ($LASTEXITCODE -ne 0) { throw "Numba no pudo compilar o ejecutar el kernel CUDA." }
    & $Python -m pip freeze --all | Out-File -Encoding utf8 results/environment/pip-freeze-numba.txt
    Write-Host "Entorno Numba CUDA listo: .venv-numba" -ForegroundColor Green
} finally {
    Pop-Location
}
