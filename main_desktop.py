"""
ARIA Desktop (web UI build)
Run with: python main_desktop.py

Startup sequence:
1. Init database
2. Identify user (reuse existing or prompt via terminal on first run)
3. Start TTS engine (primary engine loads in background; Kokoro is always the fallback)
4. Start face analyzer (skipped gracefully if no camera); enroll a
   reference face for identity verification if this user has none yet
5. Start wake-word listener thread (puts recognised text onto voice queue)
6. Start Flask in a daemon thread on a random local port
7. Open a borderless pywebview window at http://127.0.0.1:<port>
8. On close: stop mic, stop camera, shut down TTS
"""


import os
import sys

# ── Auto-relaunch inside .venv if running with the wrong Python ──────────────
# Lets you run `python main_desktop.py` from any terminal regardless of which
# Python is active — it will silently restart itself with the venv Python if
# the current interpreter isn't the venv one.
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_HERE, ".venv", "Scripts", "python.exe")
if (
    os.path.exists(_VENV_PYTHON)
    and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON)
):
    import subprocess
    result = subprocess.run([_VENV_PYTHON] + sys.argv)
    sys.exit(result.returncode)
# ─────────────────────────────────────────────────────────────────────────────

import socket
import threading
import time
from datetime import datetime


# Line-buffer stdio: with output redirected (logs, IDE terminals), Python
# block-buffers stdout and the entire voice pipeline's prints sat invisible
# in an 8KB buffer — the app looked deaf while working perfectly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except Exception:
        pass

import webview

import app_web
import database
import network_state
import patterns
import power_state
import reminders
import voice
import ui_hud


# ── helpers ────────────────────────────────────────────────────────────

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get_time_of_day() -> str:
    h = datetime.now().hour
    if 5  <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"


def _wait_for_flask(port: int, timeout: float = 8.0):
    """Block until Flask accepts connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _identify_user() -> tuple[int, str]:
    existing = database.get_first_user()
    if existing:
        database.update_user_last_seen(existing["id"])
        print(f"[main] Welcome back, {existing['name']} (id={existing['id']})")
        return existing["id"], existing["name"]

    print("[main] First run — what should ARIA call you?")
    try:
        name = input("Your name: ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    if not name:
        name = "Friend"
    user_id = database.get_or_create_user(name)
    print(f"[main] Created user '{name}' (id={user_id})")
    return user_id, name


_last_cmd: tuple | None = None   # (normalized text, wall time) — duplicate suppressor


def _voice_callback(result: dict):
    """Called from the listen_continuous thread with each recognised utterance.

    Analyses the utterance's acoustics HERE (off the request path — librosa's
    pyin takes a second or two) so /send can fuse real pitch/speed/voice-mood
    instead of the zeros that previously made voice a dead input to fusion.
    """
    # No _wake_enabled gate here anymore: the wake-word requirement is now
    # enforced inside voice.py's listen loop (voice._wake_required), so the
    # settings toggle switches modes instead of muting voice input entirely.
    text = (result.get("text") or "").strip()
    if not text:
        return

    # Loop breaker: an identical voice command within 12s is an echo artifact
    # or feedback loop, never a real request — a human doesn't repeat the
    # exact sentence twice in ten seconds and expect two answers.
    global _last_cmd
    now = time.time()
    key = text.lower().strip(" ,.!?")
    if _last_cmd and _last_cmd[0] == key and now - _last_cmd[1] < 12.0:
        print(f"[main] duplicate voice command within 12s — dropped: '{text}'")
        return
    _last_cmd = (key, now)

    print(f"[main] voice → '{text}'")

    features = {"pitch": 0.0, "speaking_speed": 0.0, "pause_ratio": 0.0, "mood": "calm"}
    audio_raw = result.get("audio_raw")
    if audio_raw:
        try:
            voice.turn_log("Voice feature analysis (librosa pyin) started")
            f = voice.analyze_voice_features(audio_raw, text)
            f["mood"] = voice.classify_voice_mood(
                f.get("pitch", 0.0), f.get("speaking_speed", 0.0), f.get("pause_ratio", 0.0)
            )
            features = f
            print(f"[main] voice features: pitch={f['pitch']}Hz "
                  f"speed={f['speaking_speed']}w/s mood={f['mood']}")
        except Exception as e:
            print(f"[main] voice feature analysis failed: {e}")

    voice.turn_log(f"Queued for UI (frontend /poll picks it up, then calls /send): '{text}'")
    app_web._voice_queue.put({"text": text, "features": features})


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("[main] Starting ARIA (web UI)…")

    database.init_db()

    print("[main] Starting power-state monitor…")
    power_state.start()

    print("[main] Starting network-state monitor…")
    network_state.start()

    user_id, user_name = _identify_user()
    app_web._user_id   = user_id
    app_web._user_name = user_name

    # print("[main] Starting telemetry HUD overlay...")
    # ui_hud.start_hud(user_id)

    engine_name = voice.TTS_ENGINE.capitalize()
    print(f"[main] Starting TTS (primary engine: {engine_name}; loads in background)…")
    voice.init_tts()

    def _deferred_start():
        time.sleep(2.0)
        print("[main] Starting face analyzer (deferred)…")
        try:
            from face import FaceAnalyzer, ENROLLED_FACES_DIR
            face_analyzer = FaceAnalyzer()
            if face_analyzer.start():
                # One-time identity enrollment
                enroll_path = os.path.join(ENROLLED_FACES_DIR, f"user_{user_id}.jpg")
                if os.path.exists(enroll_path):
                    face_analyzer.set_enrollment_path(enroll_path)
                else:
                    print("[main] No enrolled face on file for this user — enrolling now (look at the camera)...")
                    face_analyzer.enroll(enroll_path)
            app_web._face_analyzer = face_analyzer
        except Exception as e:
            print(f"[main] Face analyzer unavailable: {e}")

        print("[main] Starting wake-word listener (deferred)… [DISABLED]")
        # threading.Thread(
        #     target=voice.listen_continuous,
        #     args=(_voice_callback,),
        #     kwargs={"wake_word": "aria"},
        #     daemon=True,
        # ).start()

    threading.Thread(target=_deferred_start, daemon=True).start()

    print("[main] Starting task reminders…")
    reminders.start(user_id)

    # Warm up the personal KNN mood classifier from existing history so a
    # returning user gets learned predictions from the very first turn.
    threading.Thread(
        target=patterns.maybe_train_knn, args=(user_id, None), daemon=True
    ).start()

    port = _find_free_port()
    print(f"[main] Starting Flask on port {port}…")

    def _run_flask():
        app_web.app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

    threading.Thread(target=_run_flask, daemon=True).start()

    if not _wait_for_flask(port):
        print("[main] Flask did not start in time — aborting")
        sys.exit(1)

    print(f"[main] Flask ready at http://127.0.0.1:{port}")

    def _on_closed():
        print("[main] Window closed — shutting down…")
        voice.stop_continuous_listening()
        reminders.stop()
        voice.shutdown_tts()
        if hasattr(app_web, '_face_analyzer') and app_web._face_analyzer:
            app_web._face_analyzer.stop()
        print("[main] Goodbye.")

    class _Api:
        """Minimal JS-callable API exposed to the webview page."""
        _win = None  # underscore prefix → pywebview skips it during JS API reflection

        def toggle_fullscreen(self):
            if self._win:
                self._win.toggle_fullscreen()

        def close_window(self):
            if self._win:
                self._win.destroy()

    api = _Api()

    window = webview.create_window(
        title="ARIA",
        url=f"http://127.0.0.1:{port}",
        width=1400,
        height=900,
        fullscreen=True,
        frameless=True,
        background_color="#f3ebdc",
        min_size=(800, 600),
        js_api=api,
    )
    api._win = window
    window.events.closed += _on_closed

    def _inject_shortcuts():
        window.evaluate_js(
            "document.addEventListener('keydown',function(e){"
            "if(e.key==='F11'){e.preventDefault();"
            "window.pywebview.api.toggle_fullscreen();}"
            "});"
        )

    window.events.loaded += _inject_shortcuts

    print("[main] Opening window (fullscreen — press F11 to toggle)…")
    webview.start(debug=True)


if __name__ == "__main__":
    main()
