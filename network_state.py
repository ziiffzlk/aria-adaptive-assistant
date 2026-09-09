"""
ARIA Desktop - Network State Module
Tracks whether the machine has real internet connectivity, so the app can
degrade gracefully (offline_router.py) instead of silently freezing on a
Groq call that will never return. Only Groq's open-ended generation
genuinely needs the network — everything else (STT, TTS, face, DB) already
runs 100% locally.
"""
import socket
import threading
import time

NETWORK_CHECK_INTERVAL_S = 15
# A fast, reliable, low-overhead reachability probe: raw TCP connect to a
# well-known DNS resolver, not an HTTP request — this answers "is there a
# route to the internet at all", which is what actually determines whether
# a Groq call could ever succeed, without the extra DNS+TLS overhead of
# probing Groq's own endpoint on every check.
_PROBE_HOST = "8.8.8.8"
_PROBE_PORT = 53
_PROBE_TIMEOUT_S = 2.0

_lock = threading.Lock()
_is_online = True  # optimistic default until the first real check completes
_started = False


def is_online() -> bool:
    with _lock:
        return _is_online


def _check_once() -> bool:
    try:
        with socket.create_connection((_PROBE_HOST, _PROBE_PORT), timeout=_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def _monitor_loop():
    global _is_online
    while True:
        new_state = _check_once()
        with _lock:
            changed = new_state != _is_online
            _is_online = new_state
        if changed:
            print(f"[network] Connectivity changed: {'online' if new_state else 'OFFLINE'}")
        time.sleep(NETWORK_CHECK_INTERVAL_S)


def start():
    """Read initial connectivity, log it, and start the background monitor."""
    global _started, _is_online
    with _lock:
        if _started:
            return
        _started = True
        _is_online = _check_once()
    print(f"[network] Monitor started (checking every {NETWORK_CHECK_INTERVAL_S}s). "
          f"Initial state: {'online' if _is_online else 'OFFLINE'}")
    threading.Thread(target=_monitor_loop, daemon=True).start()


if __name__ == "__main__":
    print("Online:", _check_once())
