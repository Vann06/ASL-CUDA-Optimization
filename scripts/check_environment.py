"""Registra el entorno y ejecuta un smoke test de PyTorch.

No simula CUDA: si se solicita ``--device cuda`` y CUDA no está disponible,
termina con código distinto de cero. El reporte omite usuario, hostname y rutas.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = (3, 11)
EXPECTED_TORCH = "2.6.0"
EXPECTED_TORCHVISION = "0.21.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Dispositivo para el smoke test. 'cuda' nunca cae silenciosamente a CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta JSON opcional, por ejemplo results/environment/hardware.json.",
    )
    return parser.parse_args()


def base_version(version: str | None) -> str | None:
    return version.split("+", 1)[0] if version else None


def cpu_name() -> str:
    candidates = [
        platform.processor(),
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
        platform.uname().processor,
    ]
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            candidates.insert(0, result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return next((value.strip() for value in candidates if value and value.strip()), "desconocido")


def ram_bytes() -> int | None:
    if platform.system() == "Windows":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def nvidia_smi_gpus() -> tuple[bool, list[dict[str, str]]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False, []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return True, []

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, separator, driver = line.partition(",")
        gpus.append(
            {
                "name": name.strip(),
                "driver_version": driver.strip() if separator else "desconocido",
            }
        )
    return True, gpus


def write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(f"Reporte guardado en: {output}")


def main() -> int:
    args = parse_args()
    total_ram = ram_bytes()
    smi_found, smi_gpus = nvidia_smi_gpus()
    errors: list[str] = []

    python_ok = sys.version_info[:2] == EXPECTED_PYTHON
    if not python_ok:
        errors.append(
            f"Se requiere Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}; "
            f"se encontró {platform.python_version()}."
        )

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "target_3_11_met": python_ok,
            "pip": importlib.metadata.version("pip"),
        },
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "cpu": cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "ram_bytes": total_ram,
            "ram_gib": round(total_ram / 1024**3, 2) if total_ram else None,
        },
        "nvidia_smi": {
            "found": smi_found,
            "gpus": smi_gpus,
        },
        "pytorch": {},
        "smoke_test": {
            "requested_device": args.device,
            "status": "not_run",
        },
        "errors": errors,
    }

    try:
        import torch
    except Exception as exc:  # registra también fallos de DLL, no solo ImportError
        errors.append(f"No se pudo importar torch: {type(exc).__name__}: {exc}")
        report["pytorch"] = {"torch_imported": False}
        report["smoke_test"]["status"] = "failed"
        write_report(report, args.output)
        return 1

    torch_version = str(torch.__version__)
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
        errors.append("No se encontró instalado el paquete torchvision.")
    cuda_compiled = torch.version.cuda
    cuda_available = bool(torch.cuda.is_available())

    if base_version(torch_version) != EXPECTED_TORCH:
        errors.append(f"torch debe ser {EXPECTED_TORCH}; se encontró {torch_version}.")
    if base_version(torchvision_version) != EXPECTED_TORCHVISION:
        errors.append(
            f"torchvision debe ser {EXPECTED_TORCHVISION}; se encontró {torchvision_version}."
        )
    if cuda_compiled not in (None, "12.6"):
        errors.append(
            f"La rueda debe ser CPU o CUDA 12.6; torch fue compilado con CUDA {cuda_compiled}."
        )

    torch_gpus: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            torch_gpus.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_bytes": int(properties.total_memory),
                }
            )

    report["pytorch"] = {
        "torch_imported": True,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "cuda_compiled": cuda_compiled,
        "cuda_available": cuda_available,
        "gpu_count": len(torch_gpus),
        "gpus": torch_gpus,
    }

    if args.device == "cuda" and not cuda_available:
        errors.append(
            "Se solicitó CUDA, pero torch.cuda.is_available() es False. "
            "No se ejecutará el smoke test en CPU como sustituto."
        )
        report["smoke_test"].update(
            {"selected_device": None, "status": "failed"}
        )
        write_report(report, args.output)
        return 1

    selected = "cuda" if args.device == "cuda" or (args.device == "auto" and cuda_available) else "cpu"
    try:
        device = torch.device(selected)
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        right = left.transpose(0, 1)
        result = left @ right
        if result.shape != (4, 4) or not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("La operación produjo forma o valores inválidos.")
        if selected == "cuda":
            torch.cuda.synchronize()
        report["smoke_test"].update(
            {
                "selected_device": selected,
                "operation": "matmul 4x4",
                "result_shape": list(result.shape),
                "checksum": float(result.sum().item()),
                "status": "passed",
            }
        )
    except Exception as exc:
        errors.append(f"Falló el smoke test en {selected}: {type(exc).__name__}: {exc}")
        report["smoke_test"].update(
            {"selected_device": selected, "status": "failed"}
        )

    write_report(report, args.output)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
