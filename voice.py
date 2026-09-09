"""
ARIA Desktop - Voice Module
Speech recognition, acoustic feature extraction, language detection,
and a configurable text-to-speech pipeline.

Engine order is controlled by the ARIA_TTS_ENGINE env var:
  "chatterbox" → Chatterbox (GPU-accelerated, expressive) → Kokoro → edge-tts → pyttsx3
  "kokoro"     → Kokoro (offline ONNX) → edge-tts → pyttsx3  [default]

Set ARIA_TTS_ENGINE=chatterbox to switch.  Chatterbox falls back to Kokoro
automatically on every CUDA out-of-memory error and disables itself after 3
consecutive OOM hits so the rest of the pipeline is unaffected.
"""

import atexit
import io
import queue
import os
import re
import threading
import time
import wave
import uuid
import collections

import numpy as np
import librosa
import speech_recognition as sr
import network_state

import power_state

def clean_for_tts(text: str) -> str:
    """
    Strips raw LaTeX delimiters, Markdown symbols, and action tags so 
    Kokoro / Chatterbox only speaks clean, natural dialogue.
    """
    # 1. Remove action tags
    text = re.sub(r'\[TOOL:[^\]]*\]', '', text)

    # 2. Replace complex display math / matrix blocks with a spoken reference
    text = re.sub(r'\$\$\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}\$\$', 'as shown in the matrix on screen,', text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', 'as shown in the formula,', text)
    
    # 3. Strip multi-line code blocks and replace with a spoken cue
    text = re.sub(r'```[a-zA-Z0-9_-]*\n[\s\S]*?\n```', ' as shown in the code snippet on screen, ', text)

    # 4. Strip inline backticks (e.g., `var_name` -> "var_name")
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # 5. Convert common LaTeX math operators to spoken English
    replacements = {
        r'\\le': ' less than or equal to ',
        r'\\ge': ' greater than or equal to ',
        r'\\neq': ' not equal to ',
        r'\\times': ' times ',
        r'\\cdot': ' dot ',
        r'\\pm': ' plus or minus ',
        r'\\approx': ' approximately ',
    }
    for pattern, spoken in replacements.items():
        text = re.sub(pattern, spoken, text)

    # 4. Handle inline math: strip dollar signs, subscripts, and exponents
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)  # Extract \text{...}
    text = re.sub(r'(\w+)\^T', r'\1 transpose', text)   # Turn c^T into "c transpose"
    text = re.sub(r'(\w+)_(\w+|\d+)', r'\1 \2', text)  # Turn x_1 into "x 1", s_i into "s i"
    text = re.sub(r'\$', '', text)                     # Remove all $ symbols
    text = re.sub(r'\\[a-zA-Z]+', '', text)            # Remove remaining \backslash commands
    text = re.sub(r'[{}\[\]\(\)]', ' ', text)          # Strip braces and brackets

    # 5. Strip Markdown characters (headers, bold, italics, stray backticks)
    text = re.sub(r'[*#`_~>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
# ----------------------------------------------------------------------
# Module state
# ----------------------------------------------------------------------

_stop_event = threading.Event()
_mic_state_lock = threading.Lock()
# silent: mic has delivered pure zeros for 10s+ (device switched/muted) —
# surfaced in the UI so the user isn't left talking to a deaf assistant.
_mic_state: dict = {"level": 0.0, "active": False, "silent": False}

# Wake-word requirement. False (default): ARIA responds to ANY clear speech —
# an "Aria" prefix is still recognised and stripped when present. True: only
# utterances containing the wake word are answered. Toggleable live from the
# UI settings panel via /settings/wake.
_wake_required: bool = os.environ.get("ARIA_REQUIRE_WAKE", "0") != "0"


def set_wake_required(value: bool):
    global _wake_required
    _wake_required = bool(value)
    print(f"[voice] Wake word {'REQUIRED — say Aria first' if _wake_required else 'not required — responding to all speech'}")

_whisper_model = None    # WhisperModel on success; None before load or on failure
_whisper_tried = False   # True after the first load attempt (prevents retries on failure)
_whisper_lock  = threading.Lock()

# ── per-turn timeline (t=0 at "user stopped speaking") ─────────────────
# One continuous [t=X.XXs] block per interaction so latency loss is
# attributable stage by stage across the listen/Flask/TTS threads.
_turn_t0: float | None = None

# True only while audio is physically coming out of the speakers — polled by
# the UI (via /mic_level) so captions can sync to real playback, not to when
# the AI text arrived (which can lead the voice by the whole TTS generation).
_tts_speaking: bool = False

# C5: User baseline speaking rate (words/sec), computed from their accumulated
# voice history. None until enough samples exist (see patterns.get_user_baseline_speaking_rate).
# Set by app_web._post_turn_learning() after each successful pace calibration.
_user_baseline_rate: float | None = None


def mark_turn_start():
    global _turn_t0
    _turn_t0 = time.time()


def turn_log(label: str):
    t = (time.time() - _turn_t0) if _turn_t0 is not None else 0.0
    print(f"[t={t:6.2f}s] {label}")


_playback_started_at: float = 0.0   # wall time playback began (for barge-in "t into response")
_generation_active: bool = False    # a TTS engine is mid-generate (for barge-in log)

# What ARIA has said most recently (last 2 responses, as normalized word
# sets). The mic hears her own speakers LOUDER than the user on this hardware
# (measured RMS up to 13k vs user's 450), so level thresholds cannot separate
# her voice from an interruption — but her WORDS are known. Any "interruption"
# whose transcript overlaps her own recent speech is self-echo, not the user.
_recent_speech_words: list[set] = []


def _set_current_speech(text: str):
    words = {w.strip(",.!?;:'\"").lower() for w in text.split() if w.strip(",.!?;:'\"")}
    _recent_speech_words.append(words)
    del _recent_speech_words[:-2]


def _is_self_echo(text: str) -> bool:
    """True if the transcript is mostly words ARIA herself just said."""
    words = [w.strip(",.!?;:'\"").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words or not _recent_speech_words:
        return False
    spoken = set().union(*_recent_speech_words)
    overlap = sum(1 for w in words if w in spoken) / len(words)
    return overlap >= 0.6


_speech_ended_at: float = 0.0   # when playback last stopped — echo-tail window


def _set_speaking(value: bool):
    global _tts_speaking, _playback_started_at, _speech_ended_at
    if value and not _tts_speaking:
        _playback_started_at = time.time()
    if not value and _tts_speaking:
        _speech_ended_at = time.time()
    _tts_speaking = value
    # Keep the conversation state machine in lockstep with real playback:
    # SPEAKING exactly while audio is coming out, LISTENING the moment it isn't.
    set_assistant_state("SPEAKING" if value else "LISTENING")


# ── conversation state machine + barge-in ──────────────────────────────
# What ARIA is doing right now. LISTENING is the resting state once the
# mic thread is up; THINKING is set by /send while the LLM works; SPEAKING
# while audio is physically playing. Barge-in only fires from SPEAKING.
_assistant_state: str = "IDLE"

# Playback cancellation epoch: every speak() call captures the current value,
# and handle_barge_in() bumps it. Engines compare their captured epoch before
# playing each chunk — a stale epoch means the user interrupted, so generated
# audio is abandoned and a late-arriving clip can never start playing.
_speak_epoch: int = 0


_assistant_state_changed_at: float = time.time()

# States that must never persist indefinitely — THINKING/SPEAKING are both
# "waiting on an external operation to finish" states (an LLM call, audio
# playback); an unhandled exception anywhere in that path could leave the
# state stuck, which disables the input bar (see index.html's _busy gating)
# with no way to recover short of restarting the app. LISTENING is the real
# resting state (not IDLE — IDLE is only pre-start/post-shutdown), so it's
# deliberately excluded: the watchdog must never "fix" a perfectly normal
# idle-listening session.
_WATCHDOG_ACTIVE_STATES = {"THINKING", "SPEAKING"}
_STATE_WATCHDOG_TIMEOUT_S = 45
_STATE_WATCHDOG_POLL_S = 5


def get_assistant_state() -> str:
    return _assistant_state


def set_assistant_state(state: str):
    global _assistant_state, _assistant_state_changed_at
    if state != _assistant_state:
        _assistant_state_changed_at = time.time()
    _assistant_state = state


def _state_watchdog_loop():
    """Background safety net: force-reset to LISTENING if the state machine
    has been stuck in an active (THINKING/SPEAKING) state for too long. This
    is deliberately generic — it doesn't know WHY the state got stuck, just
    that it did, so it recovers from any silently-swallowed exception in the
    request/response/playback path, not just ones we've already seen."""
    while not _stop_event.is_set():
        time.sleep(_STATE_WATCHDOG_POLL_S)
        state = _assistant_state
        if state in _WATCHDOG_ACTIVE_STATES:
            elapsed = time.time() - _assistant_state_changed_at
            if elapsed > _STATE_WATCHDOG_TIMEOUT_S:
                print(f"[watchdog] Forced state reset from stuck state: {state} "
                      f"(stuck for {elapsed:.0f}s)")
                try:
                    stop_playback()
                except Exception as e:
                    print(f"[watchdog] stop_playback during recovery failed: {e}")
                set_assistant_state("LISTENING")


def stop_playback():
    """Halt all audio output paths immediately (sounddevice + pygame)."""
    try:
        while not _playback_queue.empty():
            _playback_queue.get_nowait()
            _playback_queue.task_done()
    except Exception:
        pass
        
    try:
        import sounddevice as sd
        sd.stop()   # also unblocks any sd.wait() in the speak thread
    except Exception:
        pass
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass


def handle_barge_in(speech_started_at: float | None = None):
    """
    The user started talking over ARIA. Stop her mid-word: kill playback
    NOW (~the current 30ms frame), invalidate the speak epoch so in-flight
    TTS generation abandons its remaining chunks, and drop back to
    LISTENING so the interrupting utterance is captured as a new command.
    The interrupted response is discarded entirely — never resumed.

    speech_started_at: wall time of the FIRST frame of the interrupting
    speech streak, so the detection→silence latency is measured for real.
    """
    global _speak_epoch
    t_into = time.time() - _playback_started_at if _playback_started_at else 0.0
    gen_cancelled = _generation_active
    _speak_epoch += 1
    stop_playback()
    stopped_at = time.time()
    _set_speaking(False)
    set_assistant_state("LISTENING")
    latency_ms = (stopped_at - speech_started_at) * 1000 if speech_started_at else -1.0
    print(f"[barge-in] Interruption detected at t={t_into:.2f}s into response, "
          f"playback stopped, generation cancelled={gen_cancelled} "
          f"(speech-start→silence: {latency_ms:.0f}ms)")

# Kokoro model files live alongside this script.
_HERE = os.path.dirname(os.path.abspath(__file__))

import sys
def get_asset_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(_HERE, filename)

KOKORO_MODEL_PATH  = get_asset_path("kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = get_asset_path("voices-v1.0.bin")

# ----------------------------------------------------------------------
# Mood -> voice delivery settings
# ----------------------------------------------------------------------

MOOD_VOICE_SETTINGS = {
    "stressed": {
        "rate_multiplier": 0.85,
        "instruction": "Speak slowly and calmly, in a soothing, reassuring tone, as if helping someone who is stressed relax.",
    },
    "happy": {
        "rate_multiplier": 1.0,
        "instruction": "Speak at a natural pace with a warm, upbeat and cheerful tone.",
    },
    "sad": {
        "rate_multiplier": 0.88,
        "instruction": "Speak slowly with a gentle, soft and compassionate tone.",
    },
    "excited": {
        "rate_multiplier": 1.05,
        "instruction": "Speak slightly faster with an energetic, enthusiastic tone.",
    },
    "anxious": {
        "rate_multiplier": 0.87,
        "instruction": "Speak slowly and steadily, in a calm and reassuring tone to ease anxiety.",
    },
    "calm": {
        "rate_multiplier": 0.92,
        "instruction": "Speak in a relaxed, even, warm and natural tone.",
    },
    "calm confident": {
        "rate_multiplier": 0.95,
        "instruction": "Speak with a confident, steady and warm tone.",
    },
    "engaged": {
        "rate_multiplier": 1.0,
        "instruction": "Speak with an attentive, friendly and conversational tone.",
    },
    "frustrated": {
        "rate_multiplier": 0.88,
        "instruction": "Speak calmly and patiently, in an understanding, de-escalating tone.",
    },
    "tired": {
        "rate_multiplier": 0.85,
        "instruction": "Speak gently and softly, at a slower pace, as if being mindful of someone who is tired.",
    },
    "distracted": {
        "rate_multiplier": 0.95,
        "instruction": "Speak clearly and warmly to gently draw attention back into the conversation.",
    },
    "surprised": {
        "rate_multiplier": 1.02,
        "instruction": "Speak with a lively, curious and warm tone.",
    },
    "default": {
        "rate_multiplier": 0.92,
        "instruction": "Speak naturally and warmly.",
    },
}

# Kokoro voice selection per emotional state.
# af_* = American English female  am_* = American English male
# bf_* = British English female   bm_* = British English male
# Female voices only: ARIA has one consistent female identity across every
# engine. Chatterbox clones aria_voice_reference*.wav — currently Resemble's
# official "gen_z_female" prompt (user-chosen 2026-07-03; originals: *.bak,
# other options in voice_prompts/). Kokoro's fallback stays in the af_*
# family so an engine switch never changes gender, though the fallback
# timbre won't match the cloned voice.
KOKORO_VOICE_MAP = {
    "happy":         "af_sky",
    "excited":       "af_sky",
    "surprised":     "af_sky",
    "calm":          "af_bella",
    "calm confident":"af_bella",
    "engaged":       "af_bella",
    "distracted":    "af_bella",
    "sad":           "af_sarah",
    "tired":         "af_sarah",
    "stressed":      "af_bella",
    "anxious":       "af_bella",
    "frustrated":    "af_bella",
    "default":       "af_sky",
}

# Kokoro speed multiplier per mood  (base 1.0)
KOKORO_SPEED_MAP = {
    "excited":  1.05,
    "happy":    1.0,
    "engaged":  1.0,
    "surprised":1.02,
    "calm":     0.92,
    "calm confident": 0.95,
    "distracted": 0.95,
    "sad":      0.88,
    "tired":    0.85,
    "stressed": 0.85,
    "anxious":  0.87,
    "frustrated": 0.88,
    "default":  0.92,
}

# ----------------------------------------------------------------------
# Lightweight language markers (Yoruba / Igbo / Nigerian Pidgin)
# ----------------------------------------------------------------------

PIDGIN_MARKERS = ["abeg", "wahala", "wetin", "dey ", "abi", "shey", "sabi", "waka", "oga", "na so", "no wahala"]
YORUBA_MARKERS = ["bawo", "pele", "jowo", "se daadaa", "ese", "owo mi", "alaafia", "ekaaro", "ekaasan"]
IGBO_MARKERS   = ["biko", "kedu", "daalu", "nnoo", "ndewo", "kedu ka", "imere"]

ENGLISH_MARKERS = {
    "the", "is", "are", "you", "your", "how", "what", "when", "where", "why",
    "hello", "hi", "hey", "please", "thanks", "thank", "can", "could", "would",
    "i'm", "im", "my", "this", "that", "with", "and", "have", "has", "do", "does",
}

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "yo": "Yoruba", "ig": "Igbo", "pcm": "Nigerian Pidgin", "ar": "Arabic",
    "hi": "Hindi", "pt": "Portuguese", "it": "Italian", "zh-cn": "Chinese",
    "ru": "Russian", "sw": "Swahili", "tr": "Turkish",
}

EDGE_TTS_VOICE_MAP = {
    "stressed": "en-GB-SoniaNeural", "anxious": "en-GB-SoniaNeural",
    "sad": "en-GB-LibbyNeural", "tired": "en-GB-LibbyNeural",
    "happy": "en-IE-EmilyNeural", "excited": "en-IE-EmilyNeural",
    "frustrated": "en-GB-SoniaNeural", "calm": "en-GB-SoniaNeural",
    "default": "en-GB-SoniaNeural",
}

# ── Chatterbox engine config ───────────────────────────────────────────
# Switch the primary TTS engine by setting ARIA_TTS_ENGINE in the environment
# or a .env file.  Kokoro is always the automatic fallback.
TTS_ENGINE: str = os.environ.get("ARIA_TTS_ENGINE", "kokoro").lower()


# ----------------------------------------------------------------------
# C5 — Conversational pace matching
# ----------------------------------------------------------------------

def set_user_baseline_rate(rate: float | None):
    """C5: Update the calibrated user speaking rate (words/sec).
    Called by app_web._post_turn_learning() after pace calibration. None
    disables pace matching and falls back to pure mood-based pacing.
    """
    global _user_baseline_rate
    _user_baseline_rate = rate
    if rate is not None:
        print(f"[voice] C5 user baseline rate updated: {rate:.3f} wps")


def _map_user_rate_to_aria_speed(user_wps: float) -> float:
    """C5: Map a user speaking rate (words/sec) to ARIA's Kokoro speed multiplier.

    Linear interpolation over the measured realistic user speech range:
      1.0 wps (slow) → ARIA 0.82  (deliberate, soft)
      2.5 wps (normal) → ARIA 0.92 (ARIA's natural default)
      4.0 wps (fast) → ARIA 1.05  (slightly brisk)

    Clamped to [0.80, 1.05] so a very fast or very slow user can't push
    ARIA outside her natural character range.
    """
    # Two-segment linear interpolation.
    if user_wps <= 2.5:
        # slow end: 1.0 wps → 0.82, 2.5 wps → 0.92
        t = max(0.0, (user_wps - 1.0) / 1.5)
        speed = 0.82 + t * (0.92 - 0.82)
    else:
        # fast end: 2.5 wps → 0.92, 4.0 wps → 1.05
        t = min(1.0, (user_wps - 2.5) / 1.5)
        speed = 0.92 + t * (1.05 - 0.92)
    return round(max(0.80, min(1.05, speed)), 3)


def _compute_pace_adjusted_speed(baseline_user_rate: float | None, mood: str) -> float:
    """C5: Compute ARIA's Kokoro speed for this turn.

    Mood-based pacing always applies; when a calibrated user baseline exists,
    it sets ARIA's BASE speed and the mood multiplier is applied ON TOP of it
    (not instead of it). This preserves the expressive mood differentiation
    while shifting the overall delivery toward the user's natural pace.

    baseline_user_rate=None: return pure mood-based speed (no calibration yet).
    """
    mood_speed = KOKORO_SPEED_MAP.get((mood or "").lower(), KOKORO_SPEED_MAP["default"])
    if baseline_user_rate is None:
        return mood_speed  # not calibrated yet — pure mood pacing

    aria_base = _map_user_rate_to_aria_speed(baseline_user_rate)
    # Apply mood multiplier relative to the default speed so mood differentiation
    # is preserved proportionally: a calm-adjusted turn is still slower than an
    # excited one by the same ratio as without calibration.
    default_speed = KOKORO_SPEED_MAP["default"]  # 0.92
    mood_ratio = mood_speed / default_speed
    adjusted = aria_base * mood_ratio
    return round(max(0.75, min(1.15, adjusted)), 3)


# Exaggeration (0 = flat affect, 1 = very dramatic). Tuned per mood so the
# voice actually sounds different — not just speed-shifted text.
_CB_EXAGGERATION: dict = {
    "excited":        0.75,
    "happy":          0.65,
    "surprised":      0.65,
    "calm confident": 0.55,
    "engaged":        0.55,
    "calm":           0.50,
    "distracted":     0.50,
    "stressed":       0.55,
    "frustrated":     0.60,
    "anxious":        0.45,
    "sad":            0.35,
    "tired":          0.30,
    "default":        0.50,
}

# cfg_weight: lower = more expressive / less constrained by the default voice.
_CB_CFG: dict = {
    "excited":   0.35,
    "happy":     0.40,
    "frustrated": 0.45,
    "calm":      0.50,
    "stressed":  0.50,
    "distracted": 0.50,
    "anxious":   0.55,
    "sad":       0.60,
    "tired":     0.62,
    "default":   0.50,
}


# ----------------------------------------------------------------------
# Speech recognition
# ----------------------------------------------------------------------

# Continuous raw-level logging ("[mic] Raw audio level: …" ~4x/sec). Proves the
# mic is delivering audio at all. Silence with ARIA_MIC_DEBUG=0 once verified.
MIC_DEBUG: bool = os.environ.get("ARIA_MIC_DEBUG", "1") != "0"

# Whisper model size. Benchmarked 2026-07-03 on this machine's real audio:
# tiny.en 0.12s (accuracy degraded) / base.en 0.24s (best accuracy on this
# quiet mic) / small.en 0.82s. base.en won on BOTH speed and accuracy.
# NOT downgraded further on battery (unlike TTS — see power_state.py /
# TTSEngine.speak): base.en is already the lightest option that doesn't
# measurably hurt accuracy on this mic; dropping to tiny.en would be a real
# accuracy regression traded for a STT step that's already the cheapest
# GPU-time component in the pipeline (0.24s vs Chatterbox's multi-second
# generations). The actual battery-throttle relief comes from the TTS tier
# skip, not from a Whisper downgrade that wasn't warranted by evidence.
WHISPER_MODEL: str = os.environ.get("ARIA_WHISPER_MODEL", "base.en")

# Whisper device: "auto" (GPU first), "cuda", or "cpu". Set ARIA_WHISPER_DEVICE=cpu
# to free ~400MB VRAM for Chatterbox at the cost of slower STT.
WHISPER_DEVICE: str = os.environ.get("ARIA_WHISPER_DEVICE", "auto").lower()

# Chatterbox sentence streaming: speech starts after the first sentence is
# generated instead of the whole reply. ARIA_TTS_STREAMING=0 to disable.
TTS_STREAMING: bool = os.environ.get("ARIA_TTS_STREAMING", "1") != "0"

# Chatterbox-Turbo (ResembleAI/chatterbox-turbo, 350M): benchmarked 2026-07-03
# at 3.06s vs 6.49s for the full model on the same sentence, 2.80 vs 3.21 GB
# VRAM, 1.44s to first audio when chunked. TRADEOFF: Turbo ignores the
# exaggeration/cfg_weight mood knobs — mood tone comes only from the cloned
# voice + text. ARIA_CHATTERBOX_TURBO=0 restores the full expressive model.
CB_TURBO: bool = os.environ.get("ARIA_CHATTERBOX_TURBO", "1") != "0"

# Turbo's prepare_conditionals asserts the reference is > 5s; the classic
# model accepts the original ~4s clip.
VOICE_REF_PATH      = get_asset_path("aria_voice_reference.wav")
VOICE_REF_LONG_PATH = get_asset_path("aria_voice_reference_long.wav")


def _split_sentences(text: str) -> list[str]:
    """Split a reply into sentence chunks for streamed TTS. Fragments shorter
    than 4 words are merged with the following sentence so the model isn't
    called for stubs like 'Sure.'"""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if len(parts) <= 1:
        return [text.strip()] if text.strip() else []
    merged: list[str] = []
    for p in parts:
        if merged and len(merged[-1].split()) < 4:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def _ts() -> str:
    """HH:MM:SS.mmm timestamp for pipeline logs."""
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


# ── continuous-stream VAD capture ───────────────────────────────────────
# Capture format: webrtcvad accepts only 10/20/30ms frames at 8/16/32/48kHz.
# 16kHz is chosen because Whisper consumes 16kHz directly (no resampling).
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS    = 30
VAD_FRAME_SAMPLES = VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000   # 480
VAD_FRAME_BYTES   = VAD_FRAME_SAMPLES * 2                    # int16 mono

# Tightened 2026-07-10: webrtcvad classifies frames by SPECTRAL SHAPE, not
# loudness — it can flag steady low-level noise (fans, hum, distant TV) as
# "speech-like" even when quiet. Mode 3 was tried here first, but it clips
# soft consonants/word-onsets mid-utterance and the ambient-calibrated
# energy gate (added in the same pass) already carries the idle noise-
# rejection load on its own — so 3 was pure accuracy cost with no unique
# benefit left for idle listening. Back to 2; the barge-in path keeps its
# OWN dedicated aggr-3 VAD (BARGE_VAD_AGGR below), which earns the
# strictness with a measured 0.7% false-positive rate on her own speaker
# bleed. A single 90ms streak was also too easy for a click/cough, so both
# idle and barge-in still require a sustained run (below).
VAD_AGGRESSIVENESS = int(os.environ.get("ARIA_VAD_AGGR", "3"))              # 0..3, was 3 this session, 2 before
VAD_START_FRAMES   = int(os.environ.get("ARIA_VAD_START_FRAMES", "15"))     # ~450ms, was 12 (~360ms)
VAD_END_SILENCE_MS = int(os.environ.get("ARIA_END_SILENCE_MS", "650"))
VAD_PADDING_MS     = 300  # audio kept from just before speech started
VAD_MAX_SEGMENT_S  = 20   # hard cap so a noisy room can't grow a segment forever

# Raw amplitude gate, layered IN FRONT OF webrtcvad: calibrated once at
# startup against a few seconds of real ambient noise (see the calibration
# block in _listen_continuous_impl), then a frame only counts as speech if
# BOTH the VAD says so AND its RMS clears ambient_baseline * multiplier.
# This is the layer that catches steady background noise VAD alone misses.
AMBIENT_CALIBRATION_S  = float(os.environ.get("ARIA_AMBIENT_CALIBRATION_S", "1.5"))
ENERGY_GATE_MULTIPLIER = float(os.environ.get("ARIA_ENERGY_GATE_MULT", "3.5"))
ENERGY_GATE_MIN_RMS    = float(os.environ.get("ARIA_ENERGY_GATE_MIN_RMS", "80"))
# Sanity ceiling: caught live 2026-07-09 — a single transient during the 1.5s
# calibration window (chair shift, window-open moment) skewed the 75th
# percentile to 454 RMS, producing a 1590 RMS gate that made ARIA nearly deaf
# for an entire session (only one unusually loud utterance got through). The
# statistic is now the MEDIAN (needs >50% of frames loud to skew, not 25%),
# plus this hard ceiling so one bad calibration can never lock out real speech
# — genuine speech RMS measured throughout this app's testing tops out well
# under this value on normal (non-shouted) delivery.
ENERGY_GATE_MAX_RMS    = float(os.environ.get("ARIA_ENERGY_GATE_MAX_RMS", "320"))

# Segments shorter than this after silence-finalization are discarded before
# ever reaching Whisper — a cough or click can still slip past the frame-level
# gates above but can't sustain a real segment this long.
MIN_SEGMENT_DURATION_S = float(os.environ.get("ARIA_MIN_SEGMENT_S", "0.4"))

# Post-transcription confidence gate (faster-whisper only — Google STT
# exposes no per-result confidence and already does its own filtering).
# avg_logprob: mean log-probability of the decoded tokens (0 = certain, more
# negative = less certain). no_speech_prob: the model's own estimate the
# segment contains no speech at all; segments it believes are majority
# non-speech are dropped.
# Calibrated against this app's REAL captured audio, not a generic rule of
# thumb: a -1.0 floor rejected 10/10 genuine transcriptions of the recorded
# test utterance ("My name is Sam" et al, avg_logprob -1.07 to -1.19 on this
# mic) — regression caught it. -1.5 keeps real speech from this hardware
# while still rejecting the synthetic garbage case (-2.5) used in testing.
WHISPER_MIN_AVG_LOGPROB    = float(os.environ.get("ARIA_WHISPER_MIN_LOGPROB", "-1.5"))
WHISPER_MAX_NO_SPEECH_PROB = float(os.environ.get("ARIA_WHISPER_MAX_NOSPEECH", "0.6"))

# Barge-in discrimination: without echo cancellation, ARIA hears her own
# speakers, often LOUDER than the user (measured live 2026-07-09: her bleed
# reaches RMS 1200-13000+ vs the user's genuine over-speech at 150-450).
# Interrupting her is more disruptive than missing a normal command, so
# barge-in candidates get their OWN, stricter stack: a dedicated
# aggressiveness-3 VAD (fires on just 0.7% of her own bleed frames vs 19-27%
# for modes 1/2), a higher absolute RMS floor, and a longer sustained streak
# than idle listening requires — a stray sound must not be able to cut her off.
BARGE_MIN_RMS      = float(os.environ.get("ARIA_BARGE_MIN_RMS", "300"))       # was 150
BARGE_START_FRAMES = int(os.environ.get("ARIA_BARGE_START_FRAMES", "17"))     # ~510ms
BARGE_VAD_AGGR     = 3


class _VadSegmenter:
    """
    Frame-in, segment-out state machine around webrtcvad.

    feed(frame) is called with every 30ms frame from the always-open stream;
    it returns a finalized speech segment (raw int16 bytes) once
    VAD_END_SILENCE_MS of trailing silence follows detected speech, else None.
    A rolling pre-speech padding buffer means the first syllable isn't lost.
    This replaces speech_recognition's energy-threshold listen()/pause logic.
    """

    def __init__(self, vad, start_frames: int | None = None):
        self._vad = vad
        # Idle listening and barge-in use SEPARATE instances with different
        # thresholds — barge-in's is deliberately longer (BARGE_START_FRAMES).
        self._start_frames = start_frames if start_frames is not None else VAD_START_FRAMES
        from collections import deque
        pad_frames = max(1, VAD_PADDING_MS // VAD_FRAME_MS) + self._start_frames
        self._padding = deque(maxlen=pad_frames)
        self._voiced: list[bytes] = []
        self._in_speech = False
        self._speech_streak = 0
        self._silence_ms = 0
        self._max_frames = VAD_MAX_SEGMENT_S * 1000 // VAD_FRAME_MS

    def feed(self, frame: bytes, speech_override: bool | None = None):
        """speech_override: force the speech/non-speech decision for this frame
        (used while ARIA is speaking, where quiet 'speech' is her own bleed)."""
        if speech_override is not None:
            is_speech = speech_override
        else:
            try:
                is_speech = self._vad.is_speech(frame, VAD_SAMPLE_RATE)
            except Exception:
                is_speech = False

        if not self._in_speech:
            self._padding.append(frame)
            if is_speech:
                self._speech_streak += 1
                if self._speech_streak >= self._start_frames:
                    self._in_speech = True
                    self._silence_ms = 0
                    self._voiced = list(self._padding)   # includes the streak frames
            else:
                self._speech_streak = 0
            return None

        # in speech
        self._voiced.append(frame)
        if is_speech:
            self._silence_ms = 0
        else:
            self._silence_ms += VAD_FRAME_MS

        if self._silence_ms >= VAD_END_SILENCE_MS or len(self._voiced) >= self._max_frames:
            segment = b"".join(self._voiced)
            self._reset()
            return segment
        return None

    def speech_started(self) -> bool:
        return self._in_speech

    def _reset(self):
        self._voiced = []
        self._in_speech = False
        self._speech_streak = 0
        self._silence_ms = 0
        self._padding.clear()


def _get_whisper():
    """
    Load faster-whisper once (GPU float16 → GPU int8 → CPU int8).
    Returns the WhisperModel or None if unavailable (caller falls back to Google STT).
    Thread-safe: the first caller loads the model; all subsequent callers return immediately.
    """
    global _whisper_model, _whisper_tried
    if _whisper_tried:                  # fast path — no lock after first attempt
        return _whisper_model
    # Non-blocking: if the preload thread is mid-load (can take ~40s behind
    # Chatterbox's imports at startup), don't stall this utterance waiting —
    # return None so the caller transcribes via Google STT for now.
    if not _whisper_lock.acquire(blocking=False):
        return None
    try:
        if _whisper_tried:              # another thread finished while we waited
            return _whisper_model
        try:
            import torch
            from faster_whisper import WhisperModel
            cuda_ok = torch.cuda.is_available() and WHISPER_DEVICE != "cpu"
            candidates = []
            if cuda_ok:
                candidates += [("cuda", "float16"), ("cuda", "int8")]
            candidates += [("cpu", "int8")]
            for device, ctype in candidates:
                try:
                    print(f"[voice:whisper] Loading {WHISPER_MODEL} on {device} ({ctype})…")
                    t0 = time.time()
                    _whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=ctype)
                    print(f"[voice:whisper] {WHISPER_MODEL} ready on {device}/{ctype} ({time.time()-t0:.1f}s)")
                    # Warm-up inference on 0.5s of silence: the FIRST transcribe
                    # call compiles CUDA kernels and once took ~58s while
                    # Chatterbox loaded beside it — during which the listen loop
                    # was deaf. Pay that cost here at startup instead.
                    t0 = time.time()
                    segs, _ = _whisper_model.transcribe(
                        np.zeros(8000, dtype=np.float32), beam_size=1, language="en")
                    list(segs)  # generator — must be consumed to actually run
                    print(f"[voice:whisper] warm-up inference done ({time.time()-t0:.1f}s) — first utterance will be fast")
                    break
                except Exception as e:
                    print(f"[voice:whisper] {device}/{ctype} failed: {e}")
                    _whisper_model = None
        except ImportError:
            print("[voice:whisper] faster-whisper not installed — using Google STT")
        except Exception as e:
            print(f"[voice:whisper] Load error: {e} — using Google STT")
        _whisper_tried = True           # mark complete AFTER load attempt so callers see final state
    finally:
        _whisper_lock.release()
    return _whisper_model


def _transcribe_audio(recognizer, audio):
    """
    Whisper-first (local, fast), Google-STT-fallback transcription of an
    sr.AudioData. Returns (text_or_None, confidence: dict). Shared by
    listen_once and listen_continuous so the two paths can't drift.

    confidence = {"engine": "whisper"|"google"|"none",
                  "avg_logprob": float, "no_speech_prob": float}
    Google exposes no per-result confidence and already filters internally,
    so its results carry pass-through values and are never confidence-gated.
    """
    whisper = _get_whisper()
    if whisper is not None:
        try:
            turn_log("STT (faster-whisper) transcription started")
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # Gain-normalize quiet captures: this laptop's mic array delivers
            # speech at ~2-8% of full scale even at 100% Windows input level,
            # which Whisper hears as near-silence and hallucinates on
            # ("Thank you."). Amplifying to ~85% peak restores intelligibility.
            peak = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
            if 0.0 < peak < 0.5:
                gain = min(0.85 / peak, 25.0)
                audio_np = audio_np * gain
                print(f"[stt] {_ts()} Applied {gain:.1f}x gain (capture peak was {peak:.3f})")
            # hotwords biases decoding toward the wake word — without it,
            # Whisper reliably writes "Aria" as "area" and the wake check
            # rejected real activation attempts (observed live 2026-07-03).
            segs, _info = whisper.transcribe(audio_np, beam_size=1, language="en",
                                             hotwords="Aria")
            segs = list(segs)   # materialize once: builds text AND reads confidence
            text = " ".join(s.text for s in segs).strip()
            if segs:
                avg_logprob = sum(s.avg_logprob for s in segs) / len(segs)
                no_speech_prob = max(s.no_speech_prob for s in segs)
            else:
                # Whisper found literally nothing to segment — maximally
                # unconfident rather than silently passing through "".
                avg_logprob, no_speech_prob = -999.0, 1.0
            turn_log(f"STT transcription result received: {text!r} "
                     f"(avg_logprob={avg_logprob:.2f}, no_speech_prob={no_speech_prob:.2f})")
            conf = {"engine": "whisper", "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob}
            return (text or None), conf
        except Exception as e:
            print(f"[stt] {_ts()} Whisper error: {e} — trying Google STT")
            
    if not network_state.is_online():
        print(f"[stt] {_ts()} Offline mode: skipping Google STT fallback.")
        return None, {"engine": "none", "avg_logprob": -999.0, "no_speech_prob": 1.0}
        
    try:
        print(f"[stt] {_ts()} Sending to recognition engine: google")
        text = recognizer.recognize_google(audio)
        print(f"[stt] {_ts()} Raw transcription result: {text!r}")
        return text, {"engine": "google", "avg_logprob": 0.0, "no_speech_prob": 0.0}
    except sr.UnknownValueError:
        print(f"[stt] {_ts()} Raw transcription result: '' (engine could not understand audio)")
    except sr.RequestError as e:
        print(f"[stt] {_ts()} Recognition error: Google STT unavailable — {e}")
    except Exception as e:
        print(f"[stt] {_ts()} Recognition error: {e}")
    return None, {"engine": "none", "avg_logprob": -999.0, "no_speech_prob": 1.0}


def _passes_confidence_gate(text: str, conf: dict) -> bool:
    """Post-transcription confidence gate. Whisper results only — Google
    already does its own filtering and carries pass-through confidence."""
    # Filter known Whisper hallucinations
    t_clean = re.sub(r'[^a-z]', '', text.strip().lower())
    hallucinations = {"whoa", "ohh", "function", "amisupportedtosomethinghere", "you"}
    if t_clean in hallucinations or len(t_clean) < 2:
        print(f"[stt] {_ts()} Discarded known hallucination/noise: {text!r}")
        return False

    if conf.get("engine") != "whisper":
        return True
    if conf["avg_logprob"] < WHISPER_MIN_AVG_LOGPROB or conf["no_speech_prob"] > WHISPER_MAX_NO_SPEECH_PROB:
        print(f"[stt] {_ts()} Discarded low-confidence result: {text!r} "
              f"(avg_logprob={conf['avg_logprob']:.2f}, no_speech_prob={conf['no_speech_prob']:.2f})")
        return False
    return True


def listen_once(timeout=5, phrase_time_limit=15):
    """
    Listen on the default microphone for a single utterance and transcribe it.
    Returns dict: {text, language, duration, audio_raw}
    text is None (not missing) when recognition fails.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                print("[voice] listen_once: timed out waiting for speech")
                return {"text": None, "language": None, "duration": 0.0, "audio_raw": None}
    except OSError as e:
        print(f"[voice] listen_once: microphone unavailable ({e})")
        return {"text": None, "language": None, "duration": 0.0, "audio_raw": None}

    duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
    audio_raw = audio.get_wav_data()

    text, conf = _transcribe_audio(recognizer, audio)
    if text and not _passes_confidence_gate(text, conf):
        text = None
    language = detect_language(text) if text else None
    return {"text": text, "language": language, "duration": duration, "audio_raw": audio_raw}


# How STT actually spells "Aria" in practice — observed live: 'area' (twice
# in one session), plus the common name spelling 'Arya'. Exact matching threw
# those real activation attempts away.
_WAKE_VARIANTS = {"aria", "arya", "area", "ariya", "aria's", "arias"}


def _find_wake(text: str, wake_word: str) -> tuple[bool, str]:
    """Token-level wake-word match tolerant of STT mishearings.
    Returns (matched, command_text_after_wake_word)."""
    variants = _WAKE_VARIANTS | {wake_word.lower()}
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok.strip(" ,.!?;:'\"").lower() in variants:
            command = " ".join(tokens[i + 1:]).strip(" ,.!?")
            return True, command
    return False, ""


def _compute_rms(audio) -> float:
    """Normalize 0.0–1.0 RMS level from an sr.AudioData object (16-bit PCM)."""
    try:
        data = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32)
        if len(data) == 0:
            return 0.0
        return float(min(1.0, np.sqrt(np.mean(data ** 2)) / 8192.0))
    except Exception:
        return 0.0


def listen_continuous(callback, wake_word="aria"):
    """
    Run forever (daemon thread target), listening for the wake word.
    On trigger, invokes callback({"text": command, "audio_raw": wav_bytes}).
    Thin proof-of-life wrapper: prints from INSIDE the thread the moment it
    runs, and prints a full traceback if the loop ever dies for ANY reason —
    a daemon thread must never be able to die silently again.
    """
    print(f"[voice] {_ts()} Wake-word thread ACTUALLY started, thread alive=True")
    try:
        _listen_continuous_impl(callback, wake_word)
    except BaseException:
        import traceback
        print(f"[voice] {_ts()} WAKE-WORD THREAD DIED WITH EXCEPTION:")
        traceback.print_exc()


def _frame_rms(frame: bytes) -> float:
    data = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0


def _segment_worker(seg_queue, callback, wake_word):
    """
    Consumes finalized speech segments and does ALL slow work (Whisper,
    wake/mode checks, callback) OFF the capture thread. The old design
    transcribed inline, leaving the mic deaf for seconds per utterance —
    the prime suspect for the "7 of 10 attempts get no response" bug.
    """
    recognizer = sr.Recognizer()   # only used as the Google-STT fallback handle
    _consecutive_stt_failures = 0
    while not _stop_event.is_set():
        try:
            item = seg_queue.get(timeout=1.0)
        except Exception:
            continue
        # 4-tuple: (segment, finalized_at, captured_during_aria_speech, oww_detected)
        segment, finalized_at = item[0], item[1]
        during_speech = item[2] if len(item) > 2 else False
        oww_detected = item[3] if len(item) > 3 else False
        try:
            global _turn_t0
            _turn_t0 = finalized_at   # [t=] timeline stays anchored to end-of-speech
            audio = sr.AudioData(segment, VAD_SAMPLE_RATE, 2)
            seg_rms = _compute_rms(audio)
            seg_s = len(segment) / 2 / VAD_SAMPLE_RATE
            turn_log(f"User stopped speaking (VAD segment finalized: {seg_s:.2f}s, RMS {seg_rms:.3f}, "
                     f"queue lag {time.time()-finalized_at:.2f}s)")

            if seg_s < MIN_SEGMENT_DURATION_S:
                print(f"[vad] Discarded segment: too short ({seg_s:.2f}s)")
                continue

            if seg_rms < 0.006:
                print(f"[stt] {_ts()} Segment too quiet (RMS {seg_rms:.3f}) — skipped")
                continue

            text, conf = _transcribe_audio(recognizer, audio)
            if not text or not _passes_confidence_gate(text, conf):
                if seg_rms > 0.01:
                    _consecutive_stt_failures += 1
                    if _consecutive_stt_failures >= 3:
                        now = time.time()
                        if 'last_stt_alert' not in locals():
                            last_stt_alert = 0
                        
                        # Use the module-level variable to persist across iterations
                        global _last_stt_proactive_alert
                        try:
                            _last_stt_proactive_alert
                        except NameError:
                            _last_stt_proactive_alert = 0

                        if now - _last_stt_proactive_alert > 300:  # 5 minutes
                            print(f"[voice] {_ts()} Repeated STT failures in loud environment — triggering chat mode prompt")
                            callback({"text": "(System diagnostic: The microphone is picking up heavy sustained noise and failing to understand the user. Proactively and briefly offer to switch to text/chat mode. Do not ask for their question.)", "audio_raw": None})
                            _last_stt_proactive_alert = now
                        else:
                            print(f"[voice] {_ts()} STT failure alert suppressed by 5-minute cooldown.")
                        _consecutive_stt_failures = 0
                continue
            
            _consecutive_stt_failures = 0

            # Echo window: a segment carrying HER voice can finalize up to
            # ~1.5s AFTER playback stops (the 600ms silence tail flips the
            # state to LISTENING first) — that boundary gap re-created the
            # greeting loop on 2026-07-09. So the content check applies to
            # any segment captured during speech OR finalized shortly after.
            # NOTE: for during_speech segments, playback was already stopped
            # at the frame level the instant the barge streak cleared (see
            # _listen_continuous_impl) — this is now only a secondary check
            # on whether the transcript is real content vs her own echo that
            # slipped past the frame-level gate; it no longer gates the stop.
            echo_window = during_speech or (finalized_at - _speech_ended_at) < 1.5
            if echo_window and _is_self_echo(text):
                print(f"[voice] {_ts()} Self-echo discarded (her own words): {text!r}")
                continue

            matched, command = _find_wake(text, wake_word)
            if oww_detected and not matched:
                print(f"[voice] {_ts()} openWakeWord triggered but STT missed wake word. Forcing wake.")
                matched = True
                command = text
                
            print(f"[voice] {_ts()} Wake word check on: {text!r} — "
                  f"matched={matched} (required={_wake_required})")
            if _wake_required and not matched:
                continue
            if not matched:
                # Always-listen: whole utterance is the command; short+quiet
                # transcripts are Whisper hallucinations ('Thank you.').
                if len(text.split()) < 3 and seg_rms < 0.03:
                    print(f"[voice] {_ts()} Ignored as probable noise (short + quiet)")
                    continue
                command = text.strip(" ,.!?")
            if not command:
                continue   # bare "Aria" — the next segment will carry the command

            turn_log("Speech accepted, forwarding to /send")
            callback({"text": command, "audio_raw": audio.get_wav_data()})
        except Exception:
            import traceback
            print(f"[voice] {_ts()} segment worker error:")
            traceback.print_exc()


def _listen_continuous_impl(callback, wake_word="aria"):
    """
    Continuous-stream capture with real-time VAD and barge-in.

    ONE sounddevice input stream is opened here at startup and stays open for
    the app's entire lifetime — it is never closed/reopened between turns
    (teardown happens only on shutdown via _stop_event). 30ms frames flow
    through webrtcvad continuously, whatever state ARIA is in; finalized
    segments are handed to a worker thread so capture NEVER pauses.
    """
    import queue as _queue
    import sounddevice as sd
    import webrtcvad

    _stop_event.clear()
    try:
        from openwakeword.model import Model
        print(f"[voice] Initializing openWakeWord model...")
        oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    except Exception as e:
        print(f"[voice] Failed to initialize openWakeWord: {e}")
        oww_model = None
        
    oww_detected_this_segment = False
    
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    segmenter = _VadSegmenter(vad, start_frames=VAD_START_FRAMES)   # idle listening
    seg_queue: "_queue.Queue" = _queue.Queue()

    # Slow work happens over there; this thread only moves 30ms frames.
    threading.Thread(target=_segment_worker, args=(seg_queue, callback, wake_word),
                     daemon=True).start()
    # Warm Whisper in parallel so the first utterance transcribes fast.
    threading.Thread(target=_get_whisper, daemon=True).start()
    # Safety net: recover automatically if THINKING/SPEAKING ever gets stuck.
    threading.Thread(target=_state_watchdog_loop, daemon=True).start()

    try:
        dev = sd.query_devices(kind="input")
        print(f"[voice] Using microphone device: {dev['name']} "
              f"(capture {VAD_SAMPLE_RATE}Hz, {VAD_FRAME_MS}ms frames, VAD aggressiveness {VAD_AGGRESSIVENESS})")
    except Exception as e:
        print(f"[voice] Input device lookup failed: {e}")

    with _mic_state_lock:
        _mic_state["active"] = True
        _mic_state["level"] = 0.0
        _mic_state["silent"] = False
    set_assistant_state("LISTENING")

    # Dead-mic detector state (device switched to BT headset / muted).
    silent_since: float | None = None
    silence_warned = False
    last_level_print = 0.0
    # ARIA_VAD_COMPARE=1: while SPEAKING, log every frame's verdict from all
    # three webrtcvad aggressiveness modes side by side — one live cough/tap
    # test then yields the false-positive data for modes 1/2/3 simultaneously.
    vad_compare = os.environ.get("ARIA_VAD_COMPARE", "0") == "1"
    cmp_vads = {a: webrtcvad.Vad(a) for a in (1, 2, 3)} if vad_compare else {}
    barge_vad = webrtcvad.Vad(BARGE_VAD_AGGR)   # strict mode just for barge-in
    barge_segmenter = _VadSegmenter(barge_vad, start_frames=BARGE_START_FRAMES)  # separate state from `segmenter`
    prev_state = "IDLE"   # tracks SPEAKING transitions so barge_segmenter resets cleanly each time

    try:
        # THE stream: opened once, never torn down between turns.
        with sd.RawInputStream(samplerate=VAD_SAMPLE_RATE, blocksize=VAD_FRAME_SAMPLES,
                               dtype="int16", channels=1) as stream:

            # ── Layer 1: ambient noise floor calibration ────────────────
            # A few seconds of real room noise at startup sets the amplitude
            # gate frame-level VAD decisions must also clear (see below) —
            # this catches steady low-level noise (fans, hum, distant TV)
            # that webrtcvad's spectral classifier alone can mistake for
            # speech-like content even at very low volume.
            calib_n = max(1, int(AMBIENT_CALIBRATION_S * 1000 / VAD_FRAME_MS))
            print(f"[vad] Calibrating ambient noise floor ({AMBIENT_CALIBRATION_S:.1f}s, "
                  f"{calib_n} frames)…")
            calib_rms = []
            for _ in range(calib_n):
                cbuf, _ = stream.read(VAD_FRAME_SAMPLES)
                cframe = bytes(cbuf)
                if len(cframe) == VAD_FRAME_BYTES:
                    calib_rms.append(_frame_rms(cframe))
            # Median (not mean/75th-percentile): needs MORE than half the
            # calibration frames to be loud before it can be skewed, so a
            # single transient (chair shift, a word spoken during startup)
            # can't drag the baseline up on its own — see ENERGY_GATE_MAX_RMS
            # comment above for what happened when a percentile-based stat
            # wasn't robust enough.
            ambient_rms = float(np.median(calib_rms)) if calib_rms else 30.0
            energy_gate_rms = max(ambient_rms * ENERGY_GATE_MULTIPLIER, ENERGY_GATE_MIN_RMS)
            if energy_gate_rms > ENERGY_GATE_MAX_RMS:
                print(f"[vad] WARNING: calibrated gate {energy_gate_rms:.1f} RMS exceeds sanity "
                      f"ceiling {ENERGY_GATE_MAX_RMS:.0f} — capping so a noisy calibration moment "
                      f"can't make ARIA deaf for the whole session")
                energy_gate_rms = ENERGY_GATE_MAX_RMS
            print(f"[vad] Ambient noise floor: {ambient_rms:.1f} RMS (median of "
                  f"{len(calib_rms)} frames) -> energy gate: {energy_gate_rms:.1f} RMS "
                  f"({ENERGY_GATE_MULTIPLIER:.1f}x baseline, floor {ENERGY_GATE_MIN_RMS:.0f}, "
                  f"ceiling {ENERGY_GATE_MAX_RMS:.0f})")

            print(f"[voice] Continuous VAD listener running "
                  f"(aggressiveness {VAD_AGGRESSIVENESS}, end-of-speech: {VAD_END_SILENCE_MS}ms silence, "
                  f"idle streak: {VAD_START_FRAMES} frames/{VAD_START_FRAMES*VAD_FRAME_MS}ms, "
                  f"barge-in streak: {BARGE_START_FRAMES} frames/{BARGE_START_FRAMES*VAD_FRAME_MS}ms "
                  f"@ RMS>{BARGE_MIN_RMS:.0f})")

            while not _stop_event.is_set():
                frame_buf, overflowed = stream.read(VAD_FRAME_SAMPLES)
                frame = bytes(frame_buf)
                if len(frame) != VAD_FRAME_BYTES:
                    continue
                rms = _frame_rms(frame)
                now = time.time()

                if oww_model and prev_state != "SPEAKING":
                    frame_np = np.frombuffer(frame, dtype=np.int16)
                    prediction = oww_model.predict(frame_np)
                    if prediction and any(score > 0.5 for score in prediction.values()):
                        if not oww_detected_this_segment:
                            print(f"[voice] {_ts()} openWakeWord triggered!")
                            oww_detected_this_segment = True

                # live UI level + optional debug print
                with _mic_state_lock:
                    _mic_state["level"] = min(1.0, rms / 8192.0)
                if MIC_DEBUG and now - last_level_print >= 0.25:
                    last_level_print = now
                    print(f"[mic] {_ts()} Raw audio level: {rms:.0f} (norm {min(1.0, rms/8192.0):.3f})")

                # dead-mic detector
                if rms < 2.0:
                    if silent_since is None:
                        silent_since = now
                    elif now - silent_since > 10.0 and not silence_warned:
                        silence_warned = True
                        with _mic_state_lock:
                            _mic_state["silent"] = True
                        print(f"[mic] {_ts()} WARNING: microphone has delivered pure silence for 10s "
                              "— check Windows Settings > Sound > Input (default device may have "
                              "switched to a Bluetooth headset, or the mic is muted)")
                else:
                    silent_since = None
                    if silence_warned:
                        silence_warned = False
                        with _mic_state_lock:
                            _mic_state["silent"] = False
                        print(f"[mic] {_ts()} Microphone signal restored")

                current_state = _assistant_state
                if current_state == "SPEAKING" and prev_state != "SPEAKING":
                    barge_segmenter._reset()   # fresh start each time she begins speaking
                prev_state = current_state

                if current_state == "SPEAKING":
                    # INSTANT-STOP barge-in, frame-level gate as the primary
                    # defense. Stopping was originally deferred until a full
                    # candidate segment transcribed clean of self-echo (see
                    # git history) because at THAT time barge-in shared the
                    # lenient idle thresholds (90ms streak, low RMS floor),
                    # which her own speaker bleed cleared constantly — hence
                    # the greeting loop (2026-07-09). Since then the barge
                    # path got its own dedicated strict filter: aggr-3 VAD
                    # (measured: fires on only 0.7% of her own bleed frames),
                    # RMS>BARGE_MIN_RMS, and a 17-frame/510ms CONSECUTIVE
                    # streak — 0.007^17 chance of that firing on bleed alone.
                    # That's now strong enough to trust immediately, so we
                    # stop the instant the streak clears instead of waiting
                    # for the user to finish talking AND pause for 650ms
                    # (the old design: total silence before ANY reaction,
                    # which is not barge-in at all). The transcript/self-echo
                    # check in _segment_worker is now a secondary safety net
                    # on whether to forward the text as a command, not the
                    # gate on whether to stop her.
                    try:
                        barge_speech = barge_vad.is_speech(frame, VAD_SAMPLE_RATE)
                    except Exception:
                        barge_speech = False
                    if vad_compare:
                        verdicts = {a: v.is_speech(frame, VAD_SAMPLE_RATE) for a, v in cmp_vads.items()}
                        print(f"[vadcmp] rms={rms:5.0f} "
                              + " ".join(f"aggr{a}={'S' if s else '.'}" for a, s in sorted(verdicts.items())))
                    candidate = barge_speech and rms > BARGE_MIN_RMS
                    was_open = barge_segmenter.speech_started()
                    segment = barge_segmenter.feed(frame, speech_override=candidate)
                    if not was_open and barge_segmenter.speech_started():
                        streak_start = time.time() - (BARGE_START_FRAMES * VAD_FRAME_MS / 1000.0)
                        handle_barge_in(streak_start)
                    if segment is not None:
                        seg_queue.put((segment, time.time(), True, oww_detected_this_segment))
                        oww_detected_this_segment = False
                    continue

                # ── Layer 1 (amplitude) + Layer 2 (spectral/VAD) combined ──
                # A frame only counts as speech if BOTH the ambient-calibrated
                # energy gate AND webrtcvad agree — closes the gap where
                # steady low-level noise passes VAD's spectral check alone.
                try:
                    raw_speech = vad.is_speech(frame, VAD_SAMPLE_RATE)
                except Exception:
                    raw_speech = False
                is_speech = raw_speech and rms >= energy_gate_rms
                segment = segmenter.feed(frame, speech_override=is_speech)
                if segment is not None:
                    seg_queue.put((segment, time.time(), False, oww_detected_this_segment))
                    oww_detected_this_segment = False

    except Exception as e:
        print(f"[voice] Microphone stream error: {e} — listener stopped")
        import traceback
        traceback.print_exc()
    finally:
        set_assistant_state("IDLE")
        with _mic_state_lock:
            _mic_state["active"] = False
            _mic_state["level"] = 0.0
            _mic_state["silent"] = False

    print("[voice] Continuous listener stopped")


def stop_continuous_listening():
    """Signal listen_continuous to exit."""
    _stop_event.set()


def get_mic_state() -> dict:
    """Thread-safe snapshot of mic level/activity + TTS playback state for the
    UI's 150ms /mic_level poll (captions sync to `speaking`)."""
    with _mic_state_lock:
        state = dict(_mic_state)
    state["speaking"] = _tts_speaking
    state["state"] = _assistant_state   # IDLE / LISTENING / THINKING / SPEAKING
    return state


# ----------------------------------------------------------------------
# Acoustic feature extraction
# ----------------------------------------------------------------------

def analyze_voice_features(audio_data, text):
    """
    Extract pitch, pitch variation, speaking speed and pause ratio from raw WAV bytes.
    Returns dict: {pitch, pitch_std, speaking_speed, pause_ratio}
    """
    defaults = {"pitch": 0.0, "pitch_std": 0.0, "speaking_speed": 0.0, "pause_ratio": 0.0}
    if not audio_data:
        return defaults

    try:
        # 16 kHz: pyin only needs 65-400 Hz pitch, and analysing at the mic's
        # native 44.1 kHz made this take ~3.5s per utterance — the single
        # biggest pre-AI latency found by the timeline instrumentation.
        y, sample_rate = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
        if y.size == 0 or sample_rate == 0:
            return defaults

        duration = len(y) / sample_rate

        f0, _voiced_flag, _voiced_probs = librosa.pyin(y, fmin=65, fmax=400, sr=sample_rate)
        valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch     = float(np.mean(valid_f0)) if valid_f0.size > 0 else 0.0
        pitch_std = float(np.std(valid_f0))  if valid_f0.size > 0 else 0.0

        word_count    = len(text.split()) if text else 0
        speaking_speed = (word_count / duration) if duration > 0 else 0.0

        intervals = librosa.effects.split(y, top_db=30)
        voiced_duration = sum((end - start) for start, end in intervals) / sample_rate
        silence_duration = max(duration - voiced_duration, 0.0)
        pause_ratio = (silence_duration / duration) if duration > 0 else 0.0

        return {
            "pitch":         round(float(pitch), 2),
            "pitch_std":     round(float(pitch_std), 2),
            "speaking_speed":round(float(speaking_speed), 2),
            "pause_ratio":   round(float(pause_ratio), 3),
        }
    except Exception as e:
        print(f"[voice] analyze_voice_features failed: {e}")
        return defaults


def classify_voice_mood(pitch, speed, pause_ratio=0.0):
    """Rule-based voice mood from acoustic features."""
    if pitch > 200:
        pitch_band = "high"
    elif pitch < 120:
        pitch_band = "low"
    else:
        pitch_band = "medium"

    if speed > 3:
        speed_band = "fast"
    elif speed < 1.5:
        speed_band = "slow"
    else:
        speed_band = "medium"

    grid = {
        ("high", "fast"): "stressed",
        ("high", "slow"): "anxious",
        ("high", "medium"): "stressed",
        ("low", "slow"): "sad",
        ("low", "fast"): "calm confident",
        ("low", "medium"): "calm",
        ("medium", "fast"): "excited",
        ("medium", "medium"): "engaged",
        ("medium", "slow"): "calm",
    }
    mood = grid.get((pitch_band, speed_band), "calm")

    if pause_ratio > 0.5 and mood in ("calm", "engaged"):
        mood = "anxious"

    return mood


# ----------------------------------------------------------------------
# Language detection
# ----------------------------------------------------------------------

def _marker_match(marker: str, lowered: str) -> bool:
    """Whole-word marker match. Plain substring matching caused false language
    switches from ordinary English: 'ese' (Yoruba) is inside 'these',
    'abi' (Pidgin) is inside 'ability', 'oga' is inside 'yoga'."""
    return re.search(r"\b" + re.escape(marker.strip()) + r"\b", lowered) is not None


def _detect_language_once(text):
    """
    Stateless single-utterance detection. Returns (lang, confidence 0.0-1.0).
    Marker hits are high-confidence; langdetect results carry their own
    probability, capped low for short utterances where it's unreliable.
    """
    if not text or not text.strip():
        return "en", 0.0

    lowered = text.lower()
    n_words = len(lowered.split())

    for marker in PIDGIN_MARKERS:
        if _marker_match(marker, lowered):
            return "pcm", 0.95
    for marker in YORUBA_MARKERS:
        if _marker_match(marker, lowered):
            return "yo", 0.95
    for marker in IGBO_MARKERS:
        if _marker_match(marker, lowered):
            return "ig", 0.95

    words = re.findall(r"[a-zA-Z']+", lowered)
    if words and lowered.isascii() and any(w in ENGLISH_MARKERS for w in words):
        return "en", 0.95

    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 0
        best = detect_langs(text)[0]
        conf = float(best.prob)
        # langdetect is statistically unreliable on short text — cap its
        # confidence below the switch threshold so a two-word utterance,
        # background noise, or a partial phrase can never flip the language.
        if n_words < 4:
            conf = min(conf, 0.50)
        return best.lang, conf
    except Exception:
        return "en", 0.0


def detect_language(text):
    """Stateless detection (language code only) — kept for callers that just
    need a one-off reading. Conversation flow should use resolve_language()."""
    return _detect_language_once(text)[0]


# ── sticky conversation language ───────────────────────────────────────
# The conversation language only switches when a DIFFERENT language is
# detected confidently on 2 consecutive utterances — a single misfire
# (noise, a short phrase, one borrowed word) can no longer flip it.
LANG_SWITCH_CONFIDENCE  = 0.80
LANG_SWITCH_CONSECUTIVE = 2

_lang_state = {"current": "en", "pending": None, "pending_count": 0}
_lang_lock = threading.Lock()


def resolve_language(text):
    """
    Confidence-gated, sticky language resolution for the conversation flow.
    Returns the language ARIA should respond in for this turn.
    """
    lang, conf = _detect_language_once(text)
    with _lang_lock:
        current = _lang_state["current"]
        switching = False

        if lang == current:
            # Agreement resets any half-built case for switching away.
            _lang_state["pending"], _lang_state["pending_count"] = None, 0
        elif conf >= LANG_SWITCH_CONFIDENCE:
            if _lang_state["pending"] == lang:
                _lang_state["pending_count"] += 1
            else:
                _lang_state["pending"], _lang_state["pending_count"] = lang, 1
            if _lang_state["pending_count"] >= LANG_SWITCH_CONSECUTIVE:
                _lang_state["current"] = lang
                _lang_state["pending"], _lang_state["pending_count"] = None, 0
                switching = True
        # low-confidence disagreement: ignored entirely, pending case unchanged

        print(f"[voice] Detected lang={lang} confidence={conf:.2f} "
              f"current={current} switching={switching}"
              + (f" (pending {_lang_state['pending']} "
                 f"{_lang_state['pending_count']}/{LANG_SWITCH_CONSECUTIVE})"
                 if _lang_state["pending"] else ""))
        return _lang_state["current"]


def get_language_name(code):
    return LANGUAGE_NAMES.get(code, code.upper() if code else "English")


# ----------------------------------------------------------------------
# Kokoro TTS engine
# ----------------------------------------------------------------------

class KokoroEngine:
    """
    Wraps kokoro-onnx for high-quality offline synthesis.
    Model is loaded lazily on first speak() call (takes ~2-3s) and then
    re-used for the session. Audio is played via sounddevice so no temp
    files are needed and playback is synchronous (blocks until done).
    """

    def __init__(self):
        self._kokoro = None
        self._load_lock = threading.Lock()
        self.available = False
        # Load synchronously so it's instantly available for the startup greeting.
        self._load()

    def _load(self):
        if not (os.path.exists(KOKORO_MODEL_PATH) and os.path.exists(KOKORO_VOICES_PATH)):
            print("[voice:kokoro] Model files not found — Kokoro disabled. "
                  "Place kokoro-v1.0.onnx and voices-v1.0.bin next to voice.py.")
            return
        try:
            from kokoro_onnx import Kokoro
            with self._load_lock:
                self._kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
                self.available = True
            print("[voice:kokoro] Model loaded — Kokoro TTS ready")
        except Exception as e:
            print(f"[voice:kokoro] Failed to load model: {e}")

    def speak(self, text, mood="calm"):
        """Synthesise and play text. Blocks until playback finishes. Returns True on success."""
        if not self.available or self._kokoro is None:
            return False
            
        # Strip bracketed emotion tags so Kokoro doesn't read them aloud
        tag_cleaner = re.compile(r'\[(chuckle|laugh|sigh|cough|gasp|sniff|clear throat|sush|groan)\]', re.IGNORECASE)
        text = tag_cleaner.sub('', text).strip()
        
        if not text:
            return True # Nothing left to say after stripping tags
            
        try:
            import sounddevice as sd
            voice_name = KOKORO_VOICE_MAP.get(mood, KOKORO_VOICE_MAP["default"])
            # C5: use pace-adjusted speed (user baseline + mood ratio) rather
            # than a fixed per-mood lookup. Falls back to pure mood pacing
            # when no baseline has been calibrated yet.
            speed = _compute_pace_adjusted_speed(_user_baseline_rate, mood)
            turn_log("TTS (Kokoro) generation started")
            t0 = time.time()
            epoch = _speak_epoch
            global _generation_active
            _generation_active = True
            try:
                samples, sample_rate = self._kokoro.create(text, voice=voice_name, speed=speed, lang="en-us")

            finally:
                _generation_active = False
            turn_log(f"TTS (Kokoro) generation FINISHED (audio ready, took {time.time()-t0:.2f}s)")
            if epoch != _speak_epoch:
                turn_log("TTS (Kokoro) generated after barge-in — discarded")
                return True
            _playback_queue.put((samples, sample_rate, epoch))
            _playback_queue.put((None, None, epoch))
            turn_log("Audio playback chunk enqueued to background worker")
            return True
        except Exception as e:
            print(f"[voice:kokoro] speak() failed: {e}")
            return False


# ----------------------------------------------------------------------
# Chatterbox TTS engine (GPU-accelerated expressive synthesis)
# ----------------------------------------------------------------------

def _save_wav_i16(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write a float32 numpy audio array to a 16-bit mono PCM WAV (stdlib only)."""
    import wave
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class ChatterboxEngine:
    """
    Wraps Chatterbox TTS for expressive neural synthesis on GPU (or CPU).

    Loads the model lazily in a background thread — same pattern as KokoroEngine.
    On CUDA out-of-memory errors, empties the CUDA cache and returns False so
    TTSEngine falls back to Kokoro transparently.  After 3 consecutive OOM hits
    the engine marks itself unavailable so it stops competing for VRAM for the
    rest of the session.

    Two variants (see CB_TURBO): ChatterboxTurboTTS (350M, ~3s/reply, no mood
    knobs) or the full ChatterboxTTS (500M, ~6.5s/reply, exaggeration+cfg).
    """

    def __init__(self):
        self._model = None
        self._lock  = threading.Lock()
        self.available   = False
        self.is_turbo    = CB_TURBO
        self._oom_strikes = 0
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import torch
            import perth

            # perth.PerthImplicitWatermarker is None on Windows (native lib missing).
            # Patch it with a no-op so ChatterboxTTS.__init__ doesn't crash.
            if perth.PerthImplicitWatermarker is None:
                class _NoOpWatermarker:
                    def apply_watermark(self, wav, sample_rate=None):
                        return wav
                perth.PerthImplicitWatermarker = _NoOpWatermarker
                print("[voice:chatterbox] perth watermarker unavailable — using no-op patch")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            variant = "Turbo (350M)" if self.is_turbo else "full (500M)"
            print(f"[voice:chatterbox] Loading {variant} model on {device}…")
            with self._lock:
                if self.is_turbo:
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    self._model = ChatterboxTurboTTS.from_pretrained(device=device)
                    # Cache our voice identity once (saves ~0.1s/call and lets
                    # generate() run without re-reading the reference file).
                    # norm_loudness=False: pyloudnorm outputs float64, which
                    # crashes Turbo's mel matmul (upstream dtype bug).
                    if os.path.exists(VOICE_REF_LONG_PATH):
                        self._model.prepare_conditionals(VOICE_REF_LONG_PATH, norm_loudness=False)
                        print("[voice:chatterbox] Turbo voice identity prepared from long reference")
                    else:
                        print("[voice:chatterbox] Long voice reference missing — Turbo will use its built-in voice")
                else:
                    from chatterbox.tts import ChatterboxTTS
                    self._model = ChatterboxTTS.from_pretrained(device=device)
                self.available = True
            print(f"[voice:chatterbox] {variant} model ready on {device}")
        except ImportError:
            print("[voice:chatterbox] chatterbox-tts not installed — engine disabled")
        except Exception as e:
            print(f"[voice:chatterbox] Failed to load model: {e}")

    def speak(self, text, mood="calm") -> bool:
        """Synthesise and play text. Returns False to signal TTSEngine should fall back."""
        if not self.available or self._model is None:
            return False
        try:
            import torch
            import sounddevice as sd

            # ── MOOD-BASED TONE (full model only — Turbo ignores these knobs;
            #    its mood expression comes from the cloned voice + wording) ─────
            exaggeration = _CB_EXAGGERATION.get(mood, _CB_EXAGGERATION["default"])
            cfg_weight   = _CB_CFG.get(mood, _CB_CFG["default"])

            # ── VOICE IDENTITY (fixed — same female voice every call) ─────────
            # Turbo: conditionals were prepared once at load from the long
            # reference, so no path is passed per call. Full model: clone from
            # the reference on every generate().
            audio_prompt_path = VOICE_REF_PATH if os.path.exists(VOICE_REF_PATH) else None
            torch.manual_seed(42)   # anchors random sampling to a consistent voice character

            # Streamed sentence chunks: playback of chunk N overlaps generation
            # of chunk N+1 (sd.play is non-blocking), so speech starts after
            # the FIRST sentence is ready (~4.3s) instead of the whole reply
            # (~6.3s+). Chatterbox generates slower than realtime on this GPU,
            # so long multi-sentence replies can still pause briefly between
            # sentences — the price of starting early.
            chunks = _split_sentences(text) if TTS_STREAMING else [text]
            variant = "Chatterbox-Turbo" if self.is_turbo else "Chatterbox"
            turn_log(f"TTS ({variant}) generation started ({len(chunks)} chunk(s))")
            t0 = time.time()
            playing = False
            # Barge-in cancellation: if the user interrupts, handle_barge_in()
            # bumps the epoch — every checkpoint below abandons the response.
            epoch = _speak_epoch
            global _generation_active
            _generation_active = True
            try:
                for i, chunk in enumerate(chunks):
                    if epoch != _speak_epoch:
                        turn_log(f"TTS ({variant}) abandoned before chunk {i+1} (barge-in)")
                        return True
                    # inference_mode: measured 7.50s → 6.25s (2026-07-03).
                    # (autocast fp16 measured SLOWER on GTX 1660 Ti — 11.0s —
                    # and cudnn.benchmark gave no gain; both deliberately absent.)
                    with torch.inference_mode():
                        if self.is_turbo:
                            # Voice identity was cached at load; mood knobs are
                            # unsupported by Turbo (it warns and ignores them).
                            wav = self._model.generate(chunk, norm_loudness=False)
                        else:
                            wav = self._model.generate(
                                chunk,
                                audio_prompt_path=audio_prompt_path,  # [IDENTITY] clones aria_voice_reference.wav
                                exaggeration=exaggeration,             # [MOOD TONE] 0.30 (tired) → 0.75 (excited)
                                cfg_weight=cfg_weight,                 # [MOOD TONE] 0.35 (excited) → 0.62 (tired)
                            )
                    audio = wav.squeeze(0).cpu().numpy()
                    if i == 0:
                        turn_log(f"TTS ({variant}) first chunk ready (took {time.time()-t0:.2f}s)")
                    # Generation of this chunk may have outlived a barge-in:
                    # a stale epoch means this clip must never reach the speakers.
                    if epoch != _speak_epoch:
                        turn_log(f"TTS ({variant}) chunk {i+1} generated after barge-in — discarded")
                        return True
                        
                    _playback_queue.put((audio, self._model.sr, epoch))
                    if i == 0:
                        turn_log("Audio playback chunk enqueued to background worker")
                        
                _playback_queue.put((None, None, epoch))
                if epoch == _speak_epoch:
                    turn_log(f"TTS ({variant}) all chunks enqueued (total {time.time()-t0:.2f}s)")
            finally:
                _generation_active = False
            self._oom_strikes = 0
            return True
        except Exception as e:
            msg    = str(e).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)
            if is_oom:
                self._oom_strikes += 1
                print(f"[voice:chatterbox] CUDA OOM (strike {self._oom_strikes}/3)"
                      " — falling back to Kokoro")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if self._oom_strikes >= 3:
                    self.available = False
                    print("[voice:chatterbox] Disabled after 3 OOM errors"
                          " — Kokoro will handle TTS for this session")
            else:
                print(f"[voice:chatterbox] speak() failed: {e}")
            return False


# ----------------------------------------------------------------------
# Main TTSEngine: Chatterbox / Kokoro → edge-tts → pyttsx3
# ----------------------------------------------------------------------

class TTSEngine:
    """
    Configurable TTS pipeline.  Engine order depends on ARIA_TTS_ENGINE:
      "chatterbox" → Chatterbox → Kokoro → edge-tts → pyttsx3
      "kokoro"     → Kokoro → edge-tts → pyttsx3  (default)
    """

    def __init__(self):
        self.kokoro     = KokoroEngine()
        self.chatterbox = ChatterboxEngine() if TTS_ENGINE == "chatterbox" else None
        self.ready      = True   # kept for compatibility with main.py wait loop
        if self.chatterbox is not None:
            print("[voice] Primary engine: Chatterbox (Kokoro is fallback)")
            threading.Thread(target=self._prepare_voice_ref, daemon=True).start()
        else:
            print("[voice] Primary engine: Kokoro")

    def _prepare_voice_ref(self):
        """
        Bootstrap-only: if the voice reference WAVs are missing (fresh install),
        synthesise stand-ins with Kokoro so Chatterbox cloning still works.
        The shipped references are Resemble's official "older_female" prompt
        (user-chosen); this never overwrites existing files.
        """
        targets = {
            VOICE_REF_PATH: "Hello, I'm ARIA, your personal assistant. I'm here to help you.",
            VOICE_REF_LONG_PATH: (
                "Hello, I'm ARIA, your personal assistant. I'm here to help you with "
                "whatever you need today. I notice patterns in how you work, and I "
                "adapt to the way you like things done."
            ),
        }
        missing = {p: t for p, t in targets.items() if not os.path.exists(p)}
        if not missing:
            print("[voice:chatterbox] Voice references ready (existing)")
            return
        # Wait up to 30 s for Kokoro to finish loading.
        for _ in range(300):
            if self.kokoro.available:
                break
            time.sleep(0.1)
        else:
            print("[voice:chatterbox] Kokoro not ready after 30 s — voice reference skipped")
            return
        for path, text in missing.items():
            try:
                samples, sr = self.kokoro._kokoro.create(
                    text, voice="af_sky", speed=0.92, lang="en-us",
                )
                _save_wav_i16(path, samples, sr)
                print(f"[voice:chatterbox] Voice reference generated: {path}")
            except Exception as e:
                print(f"[voice:chatterbox] Voice reference generation failed ({path}): {e}")

    @property
    def is_primary_ready(self) -> bool:
        """True once the configured primary TTS model has finished loading."""
        if self.chatterbox is not None:
            return self.chatterbox.available
        return self.kokoro.available

    def speak(self, text, mood="calm"):
        if not text or not text.strip():
            return

        # Check network state: Chatterbox Turbo is used when ONLINE and on AC power.
        # Fallback to Kokoro when OFFLINE or on battery.
        is_online = network_state.is_online()
        on_battery = power_state.is_on_battery()
        
        if self.chatterbox is not None and self.chatterbox.available and is_online and not on_battery:
            if self.chatterbox.speak(text, mood):
                return
        elif self.chatterbox is not None and self.chatterbox.available:
            reason = "offline" if not is_online else "on battery"
            print(f"[voice] {reason} — using Kokoro instead of Chatterbox for this reply.")

        # Tier 2 (Offline or on battery): Kokoro offline ONNX
        if self.kokoro.available:
            if self.kokoro.speak(text, mood):
                return

        # Tier 3: edge-tts (online, neural)
        self._edge_tts_speak(text, mood)

    def _edge_tts_speak(self, text, mood):
        try:
            import asyncio
            import edge_tts
            import pygame

            edge_voice = EDGE_TTS_VOICE_MAP.get(mood, EDGE_TTS_VOICE_MAP["default"])

            async def _speak():
                buf = io.BytesIO()
                async for chunk in edge_tts.Communicate(text, edge_voice).stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                buf.seek(0)
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(buf, "mp3")
                try:
                    _set_speaking(True)
                    turn_log("Audio playback actually started (edge-tts)")
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                finally:
                    _set_speaking(False)

            asyncio.run(_speak())
        except Exception as e:
            print(f"[voice] edge-tts failed ({e}), falling back to pyttsx3")
            self._pyttsx3_speak(text, mood)

    def _pyttsx3_speak(self, text, mood):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            settings = MOOD_VOICE_SETTINGS.get(mood, MOOD_VOICE_SETTINGS["default"])
            base_rate = engine.getProperty("rate") or 200
            engine.setProperty("rate", int(base_rate * settings["rate_multiplier"]))
            engine.say(text)
            try:
                _set_speaking(True)
                turn_log("Audio playback actually started (pyttsx3)")
                engine.runAndWait()
            finally:
                _set_speaking(False)
        except Exception as e:
            print(f"[voice] pyttsx3 failed: {e}")

    def stop(self):
        """Halt any in-flight sounddevice playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Module-level API (unchanged surface for main.py / ui.py)
# ----------------------------------------------------------------------

_tts_engine: TTSEngine | None = None
_tts_engine_lock = threading.Lock()

# Serialises playback: /greeting and /send each spawn a speak thread, and two
# concurrent sounddevice plays cut each other off mid-word.
_speak_lock = threading.Lock()

_playback_queue = queue.Queue()

def _playback_worker():
    import sounddevice as sd
    import numpy as np
    
    current_stream = None
    current_sr = None
    last_played_epoch = -1
    
    while True:
        try:
            item = _playback_queue.get()
            if item is None:
                break
                
            audio_array, sr, epoch = item
            
            # End of utterance marker
            if audio_array is None:
                if epoch == _speak_epoch:
                    _set_speaking(False)
                _playback_queue.task_done()
                continue
                
            if epoch != _speak_epoch:
                _playback_queue.task_done()
                continue
                
            is_new_epoch = (epoch != last_played_epoch)
            buffer = [audio_array]
            total_samples = len(audio_array)
            
            # Pre-buffer 200ms if it's the start of a new epoch
            if is_new_epoch:
                _set_speaking(True)
                target_samples = int(sr * 0.200)
                while total_samples < target_samples:
                    try:
                        next_item = _playback_queue.get(timeout=0.05)
                        if next_item is None:
                            _playback_queue.put(None)
                            break
                        n_audio, n_sr, n_epoch = next_item
                        if n_audio is None:
                            _playback_queue.put(next_item)
                            break
                        if n_epoch != epoch or n_epoch != _speak_epoch:
                            _playback_queue.task_done()
                            break
                        buffer.append(n_audio)
                        total_samples += len(n_audio)
                    except queue.Empty:
                        break
                        
            audio_array = np.concatenate(buffer)
            
            if is_new_epoch:
                silence_padding = np.zeros(int(sr * 0.15), dtype=np.float32)
                audio_array = np.concatenate([silence_padding, audio_array])
                
                fade_samples = int(sr * 0.010)
                pad_samples = len(silence_padding)
                if len(audio_array) > pad_samples + fade_samples:
                    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
                    audio_array[pad_samples:pad_samples+fade_samples] *= fade_in
                last_played_epoch = epoch
                
            if current_stream is None or current_sr != sr:
                if current_stream is not None:
                    current_stream.stop()
                    current_stream.close()
                current_stream = sd.OutputStream(samplerate=sr, channels=1, dtype='float32')
                current_stream.start()
                current_sr = sr
                
            current_stream.write(audio_array)
            
            for _ in range(len(buffer)):
                _playback_queue.task_done()
                
        except Exception as e:
            print(f"[playback_worker] error: {e}")

threading.Thread(target=_playback_worker, daemon=True).start()


def _get_tts_engine() -> TTSEngine:
    global _tts_engine
    with _tts_engine_lock:
        if _tts_engine is None:
            _tts_engine = TTSEngine()
        return _tts_engine


def init_tts():
    """
    Eagerly initialise the TTS engine (and begin loading the Kokoro model in
    the background). Call early in main.py so the model is warm before the
    first greeting. No-op if already initialised.
    """
    _get_tts_engine()


import unicodedata

def speak_safe(text, mood="calm", language="en"):
    """Speak text aloud via tiered pipeline. Never raises."""
    text = clean_for_tts(text)
    if not text or not text.strip():
        return
    print(f"[voice] Speaking ({mood}, {language}): {text[:60]}{'...' if len(text) > 60 else ''}")
    
    global _speak_epoch
    if _tts_speaking or _generation_active:
        _speak_epoch += 1
        stop_playback()
        turn_log("Superseded previous response (newest reply wins)")
        
    _set_current_speech(text)   # so the mic can recognise her own words as echo
    t_lock = time.time()
    
    try:
        with _speak_lock:
            waited = time.time() - t_lock
            if waited > 0.05:
                turn_log(f"Waited {waited:.2f}s for previous speech to finish (speak lock)")
                
            engine = _get_tts_engine()
            
            # 1. Primary Engine (Chatterbox)
            try:
                # Sanitize text for UTF-8 compatibility
                utf8_text = text.encode('utf-8', 'ignore').decode('utf-8')
                if engine.chatterbox and engine.chatterbox.available:
                    if engine.chatterbox.speak(utf8_text, mood):
                        return
            except Exception as e:
                print(f"[voice] Chatterbox fallback triggered due to: {e}")
                
            # 2. Fallback Engine (Kokoro)
            try:
                if engine.kokoro and engine.kokoro.available:
                    # Unicode normalization to prevent phonemizer C-level crashes
                    normalized_text = "".join(
                        c for c in unicodedata.normalize("NFD", text)
                        if unicodedata.category(c) != "Mn"
                    )
                    if engine.kokoro.speak(normalized_text, mood):
                        return
            except Exception as e:
                print(f"[Voice Error] Kokoro fallback failed: {e}")
                
            # 3. Graceful Degradation / Silent Log
            print("[voice] All TTS engines failed or unavailable. Silent log triggered.")
            
    except Exception as e:
        print(f"[voice] speak_safe() failed entirely: {e}")
    finally:
        _set_speaking(False)  # never leave the UI thinking we're still talking

def _dispatch_speech(text: str, mood: str = "calm"):
    """
    Dynamically route speech based on network availability.
    ONLINE: ChatterboxTurboTTS (or edge-tts).
    OFFLINE (or on battery): Kokoro-82M.
    """
    engine = _get_tts_engine()
    engine.speak(text, mood)

def stream_tokens_to_voice(token_generator, mood="calm", language="en"):
    """
    Receives a generator of tokens (e.g., from local LLM).
    Accumulates tokens and splits on sentence boundaries.
    Synthesizes and speaks complete clauses immediately to minimize Time-To-First-Audio.
    """
    global _speak_epoch
    if _tts_speaking or _generation_active:
        _speak_epoch += 1
        stop_playback()
        
    def process_sentence(sentence):
        sentence = clean_for_tts(sentence)
        if not sentence.strip(): return
        print(f"[voice] Streaming clause: {sentence}")
        # Note: bypassing the outer speak() wrapper to prevent cancellation logic 
        # from killing our own pipelined chunks. We manage the lock directly.
        _set_current_speech(sentence)
        t_lock = time.time()
        with _speak_lock:
            waited = time.time() - t_lock
            if waited > 0.05:
                turn_log(f"Waited {waited:.2f}s for previous chunk to finish")
            if not _tts_speaking:
                _set_speaking(True)
            _dispatch_speech(sentence, mood)
            
    try:
        buffer = ""
        for token in token_generator:
            buffer += token
            # Check for sentence boundary: ., !, ?, or newline followed by space or end
            match = re.search(r'([.!?\n])(\s+|$)', buffer)
            if match:
                split_idx = match.end()
                sentence = buffer[:split_idx].strip()
                if sentence:
                    process_sentence(sentence)
                buffer = buffer[split_idx:]
                
        if buffer.strip():
            process_sentence(buffer.strip())
    except Exception as e:
        print(f"[voice] stream_tokens_to_voice failed: {e}")
    finally:
        _set_speaking(False)


def shutdown_tts():
    """Clean up TTS resources. Call on app shutdown."""
    global _tts_engine
    if _tts_engine:
        _tts_engine.stop()
        _tts_engine = None


atexit.register(shutdown_tts)


if __name__ == "__main__":
    print("Testing language detection...")
    print(detect_language("Hello, how are you today?"))
    print(detect_language("Bonjour, comment ça va?"))
    print(detect_language("Abeg wetin dey happen"))

    print("Testing voice mood classification...")
    print(classify_voice_mood(220, 3.5, 0.1))
    print(classify_voice_mood(100, 1.0, 0.1))
    print(classify_voice_mood(150, 2.0, 0.1))

    print("Testing TTS (Kokoro → edge-tts → pyttsx3)...")
    init_tts()
    time.sleep(3)  # let Kokoro model finish loading
    speak("Hello, this is ARIA. Kokoro TTS is now my primary voice.", "happy")
    shutdown_tts()
