import os
import sys
import time
from typing import Any, Dict

import torch


TAMIL_TEST_QUERY = "இந்தியாவின் தலைநகரம் என்ன?"
TAMIL_TEST_OUTPUT = "இந்தியா → புது தில்லி"


def configure_utf8_terminal() -> str:
    """Best-effort UTF-8 setup for Windows PowerShell/Terminal."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    if os.name == "nt":
        try:
            os.system("chcp 65001 > nul")
        except Exception:
            pass

    return (sys.stdout.encoding or "").upper()


def print_unicode_diagnostic() -> None:
    encoding = sys.stdout.encoding or "unknown"
    print("=" * 60)
    print("TERMINAL / UNICODE")
    print("=" * 60)
    print("Encoding:")
    print(f"{encoding}")
    print("\nTamil test:")
    print(f"{TAMIL_TEST_QUERY}")
    print("\nTamil output:")
    print(f"{TAMIL_TEST_OUTPUT}")
    if "UTF" not in encoding.upper():
        print("\n[WARNING] Terminal encoding is not UTF-8. Tamil input/output may render incorrectly.")
    if os.name == "nt":
        print("\n[INFO] If Tamil glyphs look broken, set your Windows Terminal/PowerShell font to a Tamil Unicode font (e.g. Noto Sans Tamil).")
    print("=" * 60)


def print_hf_auth_status() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    status = "authenticated" if token else "unauthenticated"
    print(f"[INFO] Hugging Face authentication: {status}")


def cuda_memory_gb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3),
    }


def first_parameter_info(model: Any) -> Dict[str, str]:
    try:
        param = next(model.parameters())
        return {"device": str(param.device), "dtype": str(param.dtype)}
    except Exception:
        return {"device": "unknown", "dtype": "unknown"}


def parameter_count(model: Any) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return label, result, elapsed
