
import subprocess
import platform
from pathlib import Path
from datetime import datetime

REPORT = Path("gpu_sm_diagnostic.txt")


def run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            errors="replace"
        )
        return result.stdout.strip() or result.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"Unavailable: {exc}"


def main():
    report = []
    report.append("NVIDIA GPU SM DIAGNOSTIC")
    report.append(f"Date: {datetime.now().isoformat()}")
    report.append(f"OS: {platform.platform()}")
    report.append("")

    # Basic identity and firmware information
    report.append("=== GPU IDENTITY ===")
    report.append(run_command([
        "nvidia-smi",
        "--query-gpu=name,uuid,pci.bus_id,vbios_version,driver_version",
        "--format=csv"
    ]))
    report.append("")

    # Driver-reported hardware configuration
    report.append("=== NVIDIA-SMI FULL REPORT ===")
    report.append(run_command(["nvidia-smi", "-q"]))
    report.append("")

    # CUDA runtime's view of the GPU
    report.append("=== PYTORCH CUDA PROPERTIES ===")
    try:
        import torch

        report.append(f"PyTorch version: {torch.__version__}")
        report.append(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report.extend([
                    f"GPU index: {i}",
                    f"Name: {props.name}",
                    f"SM count: {props.multi_processor_count}",
                    f"Compute capability: "
                    f"{props.major}.{props.minor}",
                    f"Total VRAM: "
                    f"{props.total_memory / (1024**3):.2f} GiB",
                    ""
                ])
    except ImportError:
        report.append(
            "PyTorch not installed. Install a CUDA-enabled build "
            "or use the NVIDIA-SMI results."
        )
    except Exception as exc:
        report.append(f"CUDA query failed: {exc}")

    report.append("")
    report.append("=== INTERPRETATION ===")
    report.append(
        "This report shows the hardware configuration exposed "
        "through standard driver interfaces. It cannot prove "
        "whether hidden SMs exist or are permanently fused off."
    )

    output = "\n".join(report)
    REPORT.write_text(output, encoding="utf-8")
    print(output)
    print(f"\nReport saved to: {REPORT.resolve()}")


if __name__ == "__main__":
    main()
