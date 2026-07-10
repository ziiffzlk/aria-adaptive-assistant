"""
ARIA mic diagnostic — completely standalone. No wake word, no AI, no app.

    .venv\\Scripts\\python.exe mic_test.py

Phase 1: opens the default microphone and prints the raw RMS level in real
         time for 15 seconds — make noise / speak and watch the number move.
Phase 2: waits for one spoken phrase (up to 10s), saves it to
         mic_test_capture.wav next to this script so you can play it back
         and hear exactly what the microphone recorded.

If Phase 1 shows levels near 0 no matter what you do, the problem is
hardware / Windows permissions / wrong device — not ARIA's code.
NOTE: close ARIA first; two programs can't hold the same mic on Windows.
"""

import os
import sys
import time

import numpy as np
import speech_recognition as sr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_test_capture.wav")
LEVEL_SECONDS = 15
BAR_FULL = 40  # width of the console level bar


def ts() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def main():
    print("=== ARIA microphone diagnostic ===\n")

    # Device inventory
    try:
        names = sr.Microphone.list_microphone_names()
        print(f"Input devices seen by PyAudio ({len(names)}):")
        for i, n in enumerate(names):
            print(f"  [{i}] {n}")
    except Exception as e:
        print(f"Could not list devices: {e}")

    try:
        mic = sr.Microphone()
    except OSError as e:
        print(f"\nFAILED to open default microphone: {e}")
        print("→ hardware / permissions / device problem (or another app holds the mic).")
        sys.exit(1)

    print(f"\nUsing default device (index {mic.device_index if mic.device_index is not None else 'system default'})")
    print(f"\n--- Phase 1: raw level for {LEVEL_SECONDS}s — speak, clap, make noise ---")

    recognizer = sr.Recognizer()
    try:
        with mic as source:
            chunk = source.CHUNK
            peak = 0.0
            t_end = time.time() + LEVEL_SECONDS
            while time.time() < t_end:
                buf = source.stream.read(chunk)
                data = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
                peak = max(peak, rms)
                bar = "#" * min(BAR_FULL, int(rms / 8192.0 * BAR_FULL))
                print(f"[mic] {ts()} Raw audio level: {rms:7.0f} |{bar:<{BAR_FULL}}|")
                # ~4 prints/sec: each CHUNK read is ~64ms at 16kHz, so read a
                # few chunks between prints to stay real-time without spam.
                for _ in range(3):
                    buf = source.stream.read(chunk)
                    data = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
                    peak = max(peak, rms)

            print(f"\nPhase 1 done. Peak RMS: {peak:.0f} "
                  f"({'GOOD — mic is receiving audio' if peak > 500 else 'VERY LOW — mic may be muted, wrong device, or blocked by Windows privacy settings'})")

            print("\n--- Phase 2: say one full sentence (10s window) ---")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"[mic] {ts()} Calibrated (energy threshold {recognizer.energy_threshold:.0f}) — speak NOW")
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print(f"[mic] {ts()} No speech detected in 10s — nothing saved.")
                print("→ If Phase 1 levels moved when you spoke, the energy threshold may be "
                      "calibrated too high (noisy room during the 1s calibration).")
                sys.exit(2)

            dur = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
            print(f"[mic] {ts()} Speech segment captured, duration={dur:.2f}s")
            with open(OUT_WAV, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"\nSaved: {OUT_WAV}")
            print("Play it back (double-click it, or):")
            print(f'  start "" "{OUT_WAV}"')
            print("If you hear your voice clearly, capture is fine and any problem is downstream (recognition/AI).")
    except OSError as e:
        print(f"\nMicrophone stream error: {e}")
        print("→ Is ARIA (or another app) currently running and holding the mic? Close it and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
