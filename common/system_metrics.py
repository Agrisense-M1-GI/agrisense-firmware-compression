"""
common/system_metrics.py
=========================
Instantaneous system-state readings (Section 6.1): CPU frequency, CPU
temperature, RAM used. Taken once per image, at the same point metrics
are otherwise collected -- not averaged over the whole processing
duration (unlike energy, which is an integrated measurement handled
separately via common/energy_uart.py).

Linux /proc and /sys only (native on Raspberry Pi OS) -- no `vcgencmd`
subprocess call, no `psutil` dependency, consistent with keeping this
repo's dependency footprint light. Each reader fails soft: if the
expected file is missing (e.g. running off-Pi during development), it
prints a warning once and returns a safe default rather than raising,
so a missing system metric never crashes the pipeline.
"""

CPU_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"
MEMINFO_PATH = "/proc/meminfo"

_warned = set()


def _warn_once(key: str, message: str):
    if key not in _warned:
        print(f"[system_metrics] WARNING: {message}")
        _warned.add(key)


def read_cpu_freq_hz() -> int:
    """Current CPU frequency in Hz (scaling_cur_freq is in kHz)."""
    try:
        with open(CPU_FREQ_PATH) as f:
            return int(f.read().strip()) * 1000
    except (FileNotFoundError, ValueError) as exc:
        _warn_once("cpu_freq", f"could not read {CPU_FREQ_PATH} ({exc}), returning 0")
        return 0


def read_cpu_temp_c() -> float:
    """Current CPU temperature in degrees Celsius (thermal_zone0 is in m°C)."""
    try:
        with open(CPU_TEMP_PATH) as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError) as exc:
        _warn_once("cpu_temp", f"could not read {CPU_TEMP_PATH} ({exc}), returning 0.0")
        return 0.0


def read_ram_used_mb() -> float:
    """Current RAM used in MB, computed as MemTotal - MemAvailable from /proc/meminfo."""
    try:
        values = {}
        with open(MEMINFO_PATH) as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(rest.strip().split()[0])  # kB
        total_kb = values["MemTotal"]
        available_kb = values["MemAvailable"]
        return (total_kb - available_kb) / 1024.0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        _warn_once("ram_used", f"could not read {MEMINFO_PATH} ({exc}), returning 0.0")
        return 0.0
