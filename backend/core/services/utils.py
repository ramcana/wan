import platform
import psutil
import time

def get_timestamp_ms():
    return int(time.time() * 1000)

def safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"

def get_system_stats():
    """Minimal system stats provider to satisfy imports."""
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_total_gb": round(vm.total / (1024**3), 2),
        "memory_used_gb": round((vm.total - vm.available) / (1024**3), 2),
        "timestamp_ms": get_timestamp_ms(),
    }