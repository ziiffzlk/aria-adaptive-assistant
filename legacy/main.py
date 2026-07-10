"""
ARIA Desktop - Main Entry Point
Run with: python main.py

Startup sequence:
1. Initialise the database
2. Identify the user (existing user, or prompt for a name on first run)
3. Start the face analyzer (camera) - app still works if no camera is found
4. Launch the main UI window
5. Generate + speak a personalised proactive greeting in the background
6. Continuous listening starts automatically after the greeting (see ui.py)
7. On window close: stop the camera, stop the mic, clean up
"""

import sys
import threading
import time
from datetime import datetime

from PyQt6.QtWidgets import QApplication, QInputDialog

import database
import brain
import voice
from face import FaceAnalyzer
from ui import ARIAWindow


def get_time_of_day():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def prompt_for_name(qt_app):
    """Show a simple name input dialog before the main window opens (first run only)."""
    name, ok = QInputDialog.getText(None, "ARIA Setup", "Welcome! What should ARIA call you?")
    if not ok or not name or not name.strip():
        name = "Friend"
    return name.strip()


def identify_user(qt_app):
    """Return (user_id, user_name) - reuses the existing user, or prompts for a name on first run."""
    existing_user = database.get_first_user()
    if existing_user:
        database.update_user_last_seen(existing_user["id"])
        print(f"[main] Welcome back, {existing_user['name']} (user_id={existing_user['id']})")
        return existing_user["id"], existing_user["name"]

    print("[main] No existing user found - prompting for a name...")
    user_name = prompt_for_name(qt_app)
    user_id = database.get_or_create_user(user_name)
    print(f"[main] Created new user '{user_name}' (user_id={user_id})")
    return user_id, user_name


def speak_greeting_when_ready(app, user_id, user_name, face_analyzer):
    """
    Runs in a background thread. Waits a short moment for the Kokoro model to
    finish loading (it starts on a daemon thread in voice.py's KokoroEngine.__init__)
    before building and speaking the personalised greeting.

    Emits app.greeting_ready (a Qt signal) rather than calling app.show_greeting
    directly — this is a plain threading.Thread with no Qt event loop of its own;
    only a real signal/slot connection safely hands control back to the GUI thread.
    """
    # Give Kokoro's background model-load thread up to 10s to finish.
    # Wait for whichever primary TTS engine is configured (Chatterbox or Kokoro).
    tts_engine = voice._get_tts_engine()
    max_wait, waited = 10.0, 0.0
    while not tts_engine.is_primary_ready and waited < max_wait:
        time.sleep(0.4)
        waited += 0.4

    if tts_engine.is_primary_ready:
        print("[main] TTS model ready — speaking greeting")
    else:
        print("[main] TTS model not ready in time — greeting will use edge-tts/pyttsx3")

    try:
        time_of_day = get_time_of_day()
        face_signals = (
            face_analyzer.get_latest_signals()
            if face_analyzer.camera_available
            else {"emotion": "neutral", "fatigue": 0.0}
        )
        greeting = brain.get_greeting(
            user_id, user_name, face_signals.get("emotion", "neutral"),
            face_signals.get("fatigue", 0.0), time_of_day,
        )
        print(f"[main] Greeting: {greeting}")
        app.greeting_ready.emit(greeting)
    except Exception as e:
        print(f"[main] Failed to generate greeting: {e}")


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    PyQt6/SIP sometimes reports an exception raised inside a Qt-invoked
    virtual method (paintEvent, eventFilter, ...) as a bare, unhelpful
    "TypeError: invalid argument to sipBadCatcherResult()" with no
    traceback - installing this hook ensures the real underlying traceback
    still gets printed if that ever happens again.
    """
    import traceback
    print("[main] UNCAUGHT EXCEPTION:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def main():
    print("[main] Starting ARIA...")
    sys.excepthook = _log_uncaught_exception

    # PyQt6 requires a QApplication instance before any QWidget (including
    # the name-prompt dialog) can be created, so it's built first, before
    # identify_user() runs.
    qt_app = QApplication(sys.argv)

    database.init_db()

    engine_name = voice.TTS_ENGINE.capitalize()
    print(f"[main] Starting TTS engine (primary: {engine_name}; loads in background)...")
    voice.init_tts()

    user_id, user_name = identify_user(qt_app)

    print("[main] Starting face analyzer...")
    face_analyzer = FaceAnalyzer()
    face_analyzer.start()  # safe no-op (camera_available=False) if no camera is found

    def on_close():
        print("[main] Shutting down...")
        voice.stop_continuous_listening()
        voice.shutdown_tts()
        face_analyzer.stop()
        print("[main] Goodbye.")

    print("[main] Building UI...")
    app = ARIAWindow(user_id, user_name, face_analyzer, on_close=on_close)
    app.show()

    threading.Thread(
        target=speak_greeting_when_ready, args=(app, user_id, user_name, face_analyzer), daemon=True
    ).start()

    # NOTE: this build's ui.py replaces wake-word-gated listening with
    # always-on continuous listening (ARIAWindow.start_auto_listen, kicked off
    # automatically after every response/greeting). Starting the old
    # voice.listen_continuous thread here as well would open a second
    # concurrent microphone stream that fights the auto-listen loop for the
    # same device - so it's intentionally left disabled.
    print("[main] Continuous listening will start automatically after the greeting.")

    print("[main] ARIA is ready.")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
