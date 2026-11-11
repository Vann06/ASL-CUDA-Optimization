[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda", "both")]
    [string]$Target = "both"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ExpectedCheckpoint = "B5589BFD7B7EF9AF3E1E41ECEE5C42BA4E9BD75A6989A092963324FC4453D19F"

function Assert-Python([string]$Python, [string]$Name) {
    if (-not (Test-Path $Python)) { throw "Falta $Name. Ejecute primero el script setup correspondiente." }
    $Version = & $Python -c "import platform; print(platform.python_version())"
    if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne "3.11.9") {
        throw "$Name debe usar Python 3.11.9; se encontró $Version."
    }
}

function Invoke-Smoke([string]$Device, [string]$Python) {
    Write-Host "Smoke test $Device" -ForegroundColor Cyan
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check falló en $Device." }
    & $Python scripts/check_environment.py --device $Device --output "results/environment/hardware-$Device.json"
    if ($LASTEXITCODE -ne 0) { throw "El entorno $Device no pasó la verificación." }
    $OutBase = "results/benchmarks_smoke/$Device"
    & $Python -m src.benchmark --device $Device --epochs 1 --batch-size 32 --repetitions 1 --num-workers 0 --seed 42 --warmup-batches 1 --max-train-batches 2 --max-eval-batches 1 --output-dir $OutBase
    if ($LASTEXITCODE -ne 0) { throw "El benchmark reducido $Device falló." }
    $Latest = Get-ChildItem $OutBase -Directory | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    & $Python scripts/validate_results.py $Latest.FullName --expect-device $Device
    if ($LASTEXITCODE -ne 0) { throw "Los resultados smoke $Device no son válidos." }
}

Push-Location $RepoRoot
try {
    $ActualCheckpoint = (Get-FileHash results/initialization/cnn2_seed42_initial.pt -Algorithm SHA256).Hash
    if ($ActualCheckpoint -ne $ExpectedCheckpoint) {
        throw "Checkpoint incorrecto: $ActualCheckpoint; esperado $ExpectedCheckpoint."
    }
    $CpuPython = Join-Path $RepoRoot ".venv-cpu\Scripts\python.exe"
    if ($Target -in @("cpu", "both")) {
        Assert-Python $CpuPython ".venv-cpu"
        & $CpuPython scripts/verify_data.py
        if ($LASTEXITCODE -ne 0) { throw "Los datos no coinciden con el manifiesto versionado." }
        & $CpuPython -m unittest -v tests.test_model tests.test_benchmark
        if ($LASTEXITCODE -ne 0) { throw "Las pruebas unitarias fallaron." }
        & $CpuPython scripts/preview_augmentation.py --device cpu
        if ($LASTEXITCODE -ne 0) { throw "La revisión de augmentation falló." }
        Invoke-Smoke "cpu" $CpuPython
    }
    if ($Target -in @("cuda", "both")) {
        $CudaPython = Join-Path $RepoRoot ".venv-cuda\Scripts\python.exe"
        Assert-Python $CudaPython ".venv-cuda"
        if ($Target -eq "cuda") {
            & $CudaPython scripts/verify_data.py
            if ($LASTEXITCODE -ne 0) { throw "Los datos no coinciden con el manifiesto versionado." }
        }
        Invoke-Smoke "cuda" $CudaPython
    }
    Write-Host "Smoke tests solicitados aprobados." -ForegroundColor Green
} finally {
    Pop-Location
}
