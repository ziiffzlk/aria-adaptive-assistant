"""
ARIA Desktop - Power State Module
Detects AC vs battery power (psutil) and, on this NVIDIA-GPU machine, queries
REAL GPU clock/power state via nvidia-smi — so ARIA reacts to confirmed
throttling rather than assuming battery always means throttled. Runs a
background monitor so main_desktop.py/voice.py can react to power changes
mid-session, not just at startup.
"""
import subprocess
import threading
import time

import psutil

POWER_CHECK_INTERVAL_S = 30

_lock = threading.Lock()
_on_battery = False
_started = False


def is_on_battery() -> bool:
    with _lock:
        return _on_battery


def _read_battery_state() -> bool:
    """True if genuinely running on battery. False for desktops (no battery
    reported at all) as well as laptops that are plugged in."""
    try:
        batt = psutil.sensors_battery()
        if batt is None:
            return False
        return not batt.power_plugged
    except Exception as e:
        print(f"[power] battery read failed: {e}")
        return False


def get_gpu_state() -> dict | None:
    """
    Query real GPU clock/power/pstate via nvidia-smi. Returns None if
    unavailable (no NVIDIA GPU, driver/tool not found, timeout). Some fields
    (e.g. power.limit) report "[N/A]" on cards that don't expose a
    configurable power limit — parsed as None rather than crashing.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.gr,clocks.max.gr,power.draw,power.limit,pstate,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) < 6:
            return None

        def _f(v):
            try:
                return float(v)
            except ValueError:
                return None  # "[N/A]" and similar

        return {
            "clock_mhz": _f(parts[0]), "clock_max_mhz": _f(parts[1]),
            "power_draw_w": _f(parts[2]), "power_limit_w": _f(parts[3]),
            "pstate": parts[4], "utilization_pct": _f(parts[5]),
        }
    except Exception as e:
        print(f"[power] nvidia-smi query failed: {e}")
        return None


def _log_snapshot(on_batt: bool):
    gpu = get_gpu_state()
    if gpu and gpu["clock_mhz"] is not None and gpu["clock_max_mhz"]:
        pct = gpu["clock_mhz"] / gpu["clock_max_mhz"] * 100
        power_str = (f"{gpu['power_draw_w']:.1f}W/{gpu['power_limit_w']:.1f}W"
                     if gpu["power_draw_w"] is not None and gpu["power_limit_w"] is not None
                     else f"{gpu['power_draw_w']:.1f}W/limit N/A" if gpu["power_draw_w"] is not None
                     else "power N/A")
        print(f"[power] on_battery={on_batt} | GPU clock={gpu['clock_mhz']:.0f}MHz "
              f"({pct:.0f}% of max {gpu['clock_max_mhz']:.0f}MHz) | power={power_str} | "
              f"pstate={gpu['pstate']} | util={gpu['utilization_pct']:.0f}%")
    else:
        print(f"[power] on_battery={on_batt} | GPU state unavailable (no nvidia-smi / no NVIDIA GPU)")


def _monitor_loop():
    global _on_battery
    while True:
        new_state = _read_battery_state()
        with _lock:
            changed = new_state != _on_battery
            _on_battery = new_state
        if changed:
            print(f"[power] Power source changed: {'battery' if new_state else 'AC'}")
            if new_state:
                print("[power] Battery detected, switching to lightweight models.")
            else:
                print("[power] AC power detected, restoring full models.")
        # Logged every cycle (not just on transition) so a live test session
        # that plugs/unplugs mid-run produces a real, continuous before/after
        # trace rather than only two isolated snapshots.
        _log_snapshot(new_state)
        time.sleep(POWER_CHECK_INTERVAL_S)


def start():
    """Read initial power state, log a startup snapshot, and start the
    background monitor. Call once from main_desktop.py at boot."""
    global _started, _on_battery
    with _lock:
        if _started:
            return
        _started = True
        _on_battery = _read_battery_state()
    print(f"[power] Monitor started (checking every {POWER_CHECK_INTERVAL_S}s). "
          f"Initial state: {'battery' if _on_battery else 'AC'}")
    _log_snapshot(_on_battery)
    threading.Thread(target=_monitor_loop, daemon=True).start()


if __name__ == "__main__":
    # Quick manual check: python power_state.py
    print("On battery:", _read_battery_state())
    print("GPU state:", get_gpu_state())
