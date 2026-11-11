[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExpected = "3.11.9"
$Venv = Join-Path $RepoRoot ".venv-cpu"

Push-Location $RepoRoot
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "No se encontró el lanzador 'py'. Instale Python $PythonExpected x64 y abra otra terminal."
    }
    $Found = & py -3.11 -c "import platform,struct; print(platform.python_version()); print(struct.calcsize('P')*8)"
    if ($LASTEXITCODE -ne 0 -or $Found.Count -lt 2) {
        throw "No se encontró Python $PythonExpected x64 mediante 'py -3.11'."
    }
    if ($Found[0].Trim() -ne $PythonExpected -or $Found[1].Trim() -ne "64") {
        throw "Se requiere Python $PythonExpected x64; se encontró $($Found -join ' / ')."
    }
    if (-not (Test-Path $Venv)) {
        & py -3.11 -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "No se pudo crear .venv-cpu." }
    }
    $Python = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw ".venv-cpu existe pero está incompleto. Elimínelo manualmente después de verificar la ruta y repita."
    }
    $VenvVersion = & $Python -c "import platform; print(platform.python_version())"
    if ($VenvVersion.Trim() -ne $PythonExpected) {
        throw ".venv-cpu existe con Python $VenvVersion. Elimínelo manualmente después de verificar la ruta y repita."
    }
    & $Python -m pip install --upgrade pip==25.0.1
    if ($LASTEXITCODE -ne 0) { throw "Falló la actualización fijada de pip." }
    & $Python -m pip install -r requirements-cpu.txt
    if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de requirements-cpu.txt." }
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check detectó conflictos." }
    & $Python scripts/check_environment.py --device cpu --output results/environment/hardware-cpu.json
    if ($LASTEXITCODE -ne 0) { throw "La verificación CPU no pasó." }
    & $Python -m pip freeze --all | Out-File -Encoding utf8 results/environment/pip-freeze-cpu.txt
    Write-Host "Entorno CPU listo: .venv-cpu" -ForegroundColor Green
} finally {
    Pop-Location
}
