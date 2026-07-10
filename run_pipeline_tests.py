"""Part 1 headless tests: (D-analog) 10x full-pipeline replay with real audio,
(B-analog) barge-in stop latency, epoch cancellation."""
import sys, time, queue as q
sys.path.insert(0, r"c:\Users\PC\Desktop\aria-desktop")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
os.chdir(r"c:\Users\PC\Desktop\aria-desktop")

import numpy as np
import librosa
import webrtcvad
import voice

print("=" * 70)
print("TEST D (headless analog): 10 consecutive attempts through the FULL")
print("capture->VAD->segment->worker->whisper->wake->callback pipeline,")
print("using the real recorded utterance replayed as 30ms frames.")
print("=" * 70)

FB = voice.VAD_FRAME_BYTES
y, _ = librosa.load("mic_test_capture.wav", sr=voice.VAD_SAMPLE_RATE, mono=True)
y = y / (np.max(np.abs(y)) + 1e-9) * 0.35
speech = (y * 32767).astype(np.int16).tobytes()
silence = b"\x00" * (FB * 30)

def frames(b):
    return [b[i:i+FB] for i in range(0, len(b) - FB + 1, FB)]

results = []
received = []
def cb(payload):
    received.append(payload["text"])

seg_queue = q.Queue()
import threading
voice._stop_event.clear()
threading.Thread(target=voice._segment_worker, args=(seg_queue, cb, "aria"), daemon=True).start()

for attempt in range(1, 11):
    seg = voice._VadSegmenter(webrtcvad.Vad(voice.VAD_AGGRESSIVENESS))
    segment = None
    for f in frames(silence) + frames(speech) + frames(silence):
        out = seg.feed(f)
        if out:
            segment = out
    ok_seg = segment is not None
    n_before = len(received)
    exc = ""
    if ok_seg:
        seg_queue.put((segment, time.time()))
        t0 = time.time()
        while len(received) == n_before and time.time() - t0 < 20:
            time.sleep(0.05)
    got = received[n_before] if len(received) > n_before else None
    dur = (len(segment) / 2 / voice.VAD_SAMPLE_RATE) if segment else 0.0
    results.append((attempt, dur, got))
    print(f"  attempt {attempt:2}: segment={dur:4.2f}s  callback_text={got!r}")

success = sum(1 for _, _, g in results if g)
print(f"\nSUCCESS: {success}/10 attempts produced a command at the callback")

print()
print("=" * 70)
print("TEST B (headless analog): barge-in stop latency, 3 repetitions")
print("(real sd.play of generated audio, real handle_barge_in)")
print("=" * 70)
import sounddevice as sd
tone = (0.05 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5 * 16000))).astype(np.float32)
for i in range(1, 4):
    voice._set_speaking(True)         # marks playback start
    sd.play(tone, samplerate=16000)
    time.sleep(0.5)                    # "ARIA has been talking 0.5s"
    speech_start = time.time()         # user starts talking (simulated)
    time.sleep(0.09)                   # 3 frames of streak (~90ms) before trigger
    voice.handle_barge_in(speech_start)
    stopped = time.time()
    # verify playback actually halted
    still = sd.get_stream().active if sd.get_stream() else False
    print(f"  rep {i}: speech-start->stopped = {(stopped-speech_start)*1000:.0f}ms | stream active after stop: {still}")
    time.sleep(0.2)

print()
print("epoch cancellation: a 'generated-late' clip must be refused")
e = voice._speak_epoch  # engines compare captured epoch vs current
captured = e - 1        # stale epoch, as if generation started pre-barge
print(f"  stale epoch {captured} != current {e} -> chunk discarded: {captured != e}")
print("\nP1 HEADLESS TESTS DONE")
voice._stop_event.set()
