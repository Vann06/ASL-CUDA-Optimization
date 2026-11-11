[CmdletBinding()]
param(
    [switch]$ConfirmFullBenchmark
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ExpectedCheckpoint = "B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F"

if (-not $ConfirmFullBenchmark) {
    throw "Ejecución cancelada. Para autorizar 10 épocas x 3 repeticiones en CPU y CUDA, repita con -ConfirmFullBenchmark."
}

function Assert-LatestSmoke([string]$Device, [string]$Python) {
    $Base = Join-Path $RepoRoot "results/benchmarks_smoke/$Device"
    $Latest = Get-ChildItem $Base -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($null -eq $Latest) { throw "Falta smoke test $Device. Ejecute scripts/run_smoke_tests.ps1 -Target both." }
    & $Python scripts/validate_results.py $Latest.FullName --expect-device $Device
    if ($LASTEXITCODE -ne 0) { throw "El smoke test $Device no es válido." }
}

function Invoke-Full([string]$Device, [string]$Python) {
    $Base = "results/benchmarks/$Device"
    $Log = "results/environment/full-$Device-console.log"
    & $Python -m src.benchmark --device $Device --epochs 10 --batch-size 32 --repetitions 3 --num-workers 0 --seed 42 --warmup-batches 2 --output-dir $Base 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "El benchmark oficial $Device falló. Revise $Log y error.json." }
    $Latest = Get-ChildItem $Base -Directory | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    & $Python scripts/validate_results.py $Latest.FullName --expect-device $Device --expect-full
    if ($LASTEXITCODE -ne 0) { throw "Los resultados oficiales $Device no pasaron la validación." }
}

Push-Location $RepoRoot
try {
    $CpuPython = Join-Path $RepoRoot ".venv-cpu\Scripts\python.exe"
    $CudaPython = Join-Path $RepoRoot ".venv-cuda\Scripts\python.exe"
    if (-not (Test-Path $CpuPython) -or -not (Test-Path $CudaPython)) {
        throw "Faltan .venv-cpu o .venv-cuda. Ejecute ambos scripts setup."
    }
    foreach ($Python in @($CpuPython, $CudaPython)) {
        $Version = & $Python -c "import platform; print(platform.python_version())"
        if ($Version.Trim() -ne "3.11.9") { throw "Ambos entornos deben usar Python 3.11.9." }
        & $Python -m pip check
        if ($LASTEXITCODE -ne 0) { throw "pip check falló en $Python." }
    }
    if ((Get-FileHash results/initialization/cnn2_seed42_initial.pt -Algorithm SHA256).Hash -ne $ExpectedCheckpoint) {
        throw "El checkpoint canónico no coincide."
    }
    & $CpuPython scripts/verify_data.py
    if ($LASTEXITCODE -ne 0) { throw "Los datos no coinciden con la referencia." }
    & $CpuPython scripts/check_environment.py --device cpu --output results/environment/hardware-cpu-official.json
    if ($LASTEXITCODE -ne 0) { throw "Entorno CPU inválido." }
    & $CudaPython scripts/check_environment.py --device cuda --output results/environment/hardware-cuda-official.json
    if ($LASTEXITCODE -ne 0) { throw "Entorno CUDA/driver inválido." }
    Assert-LatestSmoke "cpu" $CpuPython
    Assert-LatestSmoke "cuda" $CudaPython
    & nvidia-smi | Out-File -Encoding utf8 results/environment/nvidia-smi-before.txt
    Invoke-Full "cpu" $CpuPython
    Invoke-Full "cuda" $CudaPython
    & nvidia-smi | Out-File -Encoding utf8 results/environment/nvidia-smi-after.txt
    Write-Host "Benchmarks oficiales CPU y CUDA completados y validados." -ForegroundColor Green
} finally {
    Pop-Location
}
