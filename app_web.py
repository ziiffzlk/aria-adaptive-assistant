"""
ARIA Web Backend
Flask app that bridges templates/index.html to brain/voice/face/database.
Run via main_desktop.py (pywebview wrapper), not directly.
"""

import queue
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template, request

import brain
import database
import network_state
import offline_router
import patterns
import power_state
import system_actions
import voice
import fusion

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ── shared state (set by main_desktop.py before Flask starts) ──────────
_user_id:       int | None  = None
_user_name:     str         = "Friend"
_face_analyzer              = None     # FaceAnalyzer instance or None
_voice_queue:   queue.Queue = queue.Queue()
_wake_enabled:  bool        = False    # semantics: True = "Aria" prefix required; False = respond to all speech
_cam_enabled:   bool        = True     # main_desktop starts the camera at boot — keep UI/back-end consistent

# Acoustic features of the most recent voice command (set by /poll when the
# wake-word thread hands over an utterance, consumed by the /send that follows).
# Typed messages have no acoustics, so /send falls back to zeros for those.
_pending_voice_features: dict | None = None
_pending_lock = threading.Lock()


# ── helpers ────────────────────────────────────────────────────────────

def _sanitize_for_json(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _get_time_of_day() -> str:
    h = datetime.now().hour
    if 5  <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"


def _face_signals() -> dict:
    if _face_analyzer and _face_analyzer.camera_available:
        return _face_analyzer.get_latest_signals()
    # No camera in use — fail-open on identity verification (speaker_verified
    # True). This is a personal single-user desktop app, not a security
    # product: the absence of a camera must never itself block or degrade
    # normal use, per the same fail-open principle as face.FaceAnalyzer.
    return {
        "emotion": "neutral", "confidence": 0.0,
        "fatigue": 0.0, "engagement": 0.5,
        "fused_mood": "calm", "face_detected": False,
        "speaker_verified": True, "verification_distance": None,
    }


def _estimate_tts_ms(text: str, mood: str) -> int:
    """Estimate speech duration in ms — drives the caption word-reveal pacing.
    Engine-aware: Chatterbox-Turbo speaks ~3.3 words/sec (measured from its
    generated samples 2026-07-03) and ignores mood pacing; Kokoro runs the
    pace-adjusted speed formula (C5 user baseline + mood ratio).
    Using Kokoro pacing for Turbo made captions lag the voice by ~40%."""
    words = max(1, len(text.split()))
    if voice.TTS_ENGINE == "chatterbox":
        return int((words / 3.3) * 1000)
    # C5: use the same pace-adjusted speed Kokoro will actually use.
    speed = voice._compute_pace_adjusted_speed(voice._user_baseline_rate, mood)
    return int((words / (2.5 * speed)) * 1000)


def _consume_voice_features() -> dict:
    """Pop the pending acoustic features (voice turn) or return typed-turn zeros."""
    global _pending_voice_features
    with _pending_lock:
        vf = _pending_voice_features
        _pending_voice_features = None
    return vf or {"pitch": 0.0, "speaking_speed": 0.0, "pause_ratio": 0.0, "mood": "calm"}


def _post_turn_learning(user_id):
    """Background learning pass after each turn: refresh the cached behavioural
    profile, periodically retrain the personal KNN mood classifier, and
    calibrate ARIA's speech pace to the user's natural speaking rate (C5)."""
    try:
        patterns.refresh_behavioural_profile(user_id)
        total = database.get_total_conversations(user_id)
        patterns.maybe_train_knn(user_id, total)
    except Exception as e:
        print(f"[app_web] post-turn learning failed: {e}")

    # C5: pace calibration — compute user's median speaking rate and update
    # ARIA's Kokoro delivery speed. Runs every turn but get_user_baseline_speaking_rate
    # returns None (no-op) until the minimum sample count is reached.
    try:
        baseline = patterns.get_user_baseline_speaking_rate(user_id)
        if baseline is not None:
            voice.set_user_baseline_rate(baseline)
            database.log_adaptation(
                user_id, "pace_calibrated",
                f"User baseline {baseline:.3f} wps → ARIA speed adjusted "
                f"(Kokoro base: {voice._map_user_rate_to_aria_speed(baseline):.3f})",
            )
    except Exception as e:
        print(f"[app_web] pace calibration failed: {e}")



_ESCALATION_OFFLINE_TEMPLATE = (
    "I hear you, and that sounds really heavy. I'm offline right now so I can't think this "
    "through with you the way I'd like to, but please reach out to someone you trust, or a "
    "mental health professional — you don't have to carry this alone."
)


def _handle_escalation(message, fs, speaker_verified, voice_features, language):
    """
    Part B3: genuine serious distress (explicit language in THIS message, or
    a sustained severe-low-mood pattern across many real sessions) takes
    priority over everything else, including the offline branch. Uses a
    dedicated prompt (brain.build_escalation_prompt) that forbids content
    suggestions, NEVER calls extract_youtube/extract_search/extract_action
    (so even if the model slipped and emitted one, it has no way to become a
    real side effect here), and always logs 'escalation_deferred' as proof
    the system recognises its own limits — regardless of whether Groq was
    reachable.
    """
    print(f"[app_web] Serious distress signal detected for user {_user_id} — escalating, "
          f"not offering a suggestion")

    if network_state.is_online():
        try:
            prompt = brain.build_escalation_prompt(message, language)
            raw = brain.get_ai_response(prompt)
            response_text = brain.clean_response(raw)  # strips any stray marker lines as a safety net
            if not response_text:
                response_text = _ESCALATION_OFFLINE_TEMPLATE
        except Exception as e:
            print(f"[app_web] escalation prompt failed: {e}")
            response_text = _ESCALATION_OFFLINE_TEMPLATE
    else:
        response_text = _ESCALATION_OFFLINE_TEMPLATE

    try:
        database.log_adaptation(
            _user_id, "escalation_deferred",
            f"Detected serious distress signal — deferred to human/professional connection "
            f"instead of a suggestion. Message: {message[:120]!r}",
        )
    except Exception as e:
        print(f"[app_web] escalation_deferred logging failed: {e}")

    if speaker_verified:
        try:
            database.save_conversation(
                _user_id, message, response_text, "sad", "High", language,
                voice_features.get("pitch", 0.0), voice_features.get("speaking_speed", 0.0),
                fs.get("emotion", "neutral"), fs.get("fatigue", 0.0), fs.get("engagement", 0.5),
            )
            database.save_mood_reading(
                _user_id, voice_features.get("mood", "calm"), fs.get("fused_mood", "calm"),
                "sad", "sad", 0.9,
            )
        except Exception as e:
            print(f"[app_web] escalation DB write failed: {e}")

    threading.Thread(target=voice.speak, args=(response_text, "sad"), daemon=True).start()
    return jsonify(_sanitize_for_json({
        "response": response_text, "mood": "sad", "confidence": "High", "url": None,
        "searched": False, "escalated": True,
        "estimated_duration_ms": _estimate_tts_ms(response_text, "sad"),
    }))


def _handle_offline_send(message, fs, speaker_verified, voice_features):
    """
    Groq is unreachable — route through offline_router instead of freezing
    on a network call that can never return. Mood fusion still runs (voice
    and face signals are 100% local); only the TEXT-sentiment leg is skipped
    since analyze_text_mood() also needs Groq, and treated as neutral so the
    voice+face signals aren't diluted by a fake "calm" vote. Still logs to
    conversations/patterns/mood_readings when a real category (not the
    open-ended fallback refusal) was handled and the speaker is verified —
    same protection as the online path.
    """
    fused = fusion.fuse_moods(
        voice_features.get("mood", "calm"), fs.get("fused_mood", "calm"), "calm",
        voice_features.get("pitch", 0.0), fs.get("fatigue", 0.0), fs.get("engagement", 0.5),
    )
    fused_mood, fused_intensity = fused.get("mood", "calm"), fused.get("intensity", 0.5)

    response_text, category = offline_router.try_offline_response(
        message, _user_id, fused_mood, fused_intensity,
    )
    offline_hit = category is not None
    if response_text is None:
        response_text = offline_router.OFFLINE_FALLBACK_MESSAGE
        category = "offline_fallback"

    print(f"[app_web] Offline — handled as '{category}'"
          + ("" if offline_hit else " (genuine open-ended request, network needed)"))

    if speaker_verified:
        try:
            database.save_conversation(
                _user_id, message, response_text, fused_mood, "Medium", "en",
                voice_features.get("pitch", 0.0), voice_features.get("speaking_speed", 0.0),
                fs.get("emotion", "neutral"), fs.get("fatigue", 0.0), fs.get("engagement", 0.5),
            )
            database.save_mood_reading(
                _user_id, voice_features.get("mood", "calm"), fs.get("fused_mood", "calm"),
                "calm", fused_mood, fused_intensity,
            )
            patterns.update_all_patterns(
                _user_id, message, fused_mood, datetime.now().hour, "en", fs.get("emotion", "neutral"),
            )
        except Exception as e:
            print(f"[app_web] offline DB/patterns update failed: {e}")

    threading.Thread(target=voice.speak, args=(response_text, fused_mood), daemon=True).start()
    return jsonify(_sanitize_for_json({
        "response": response_text, "mood": fused_mood, "confidence": "Medium", "url": None,
        "searched": False, "offline": True,
        "estimated_duration_ms": _estimate_tts_ms(response_text, fused_mood),
    }))


def _to_visual_mood(mood: str) -> str:
    m = (mood or "").lower()
    if m in {"stressed", "anxious", "frustrated", "fearful", "fear", "angry", "disgust"}:
        return "thoughtful"
    if m in {"sad", "tired"}:
        return "tender"
    if m in {"happy", "excited", "grateful", "surprised"}:
        return "happy"
    return "calm"


# ── routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/greeting")
def greeting():
    tod = _get_time_of_day()
    fs  = _face_signals()
    try:
        text = brain.get_greeting(
            _user_id, _user_name,
            fs.get("emotion", "neutral"),
            fs.get("fatigue", 0.0),
            tod,
        )
    except Exception as e:
        print(f"[app_web] get_greeting failed: {e}")
        text = f"Good {tod}, {_user_name}. I'm here."

    def _speak_greeting():
        voice.speak_safe(text, "calm")

    threading.Thread(target=_speak_greeting, daemon=True).start()
    return jsonify({"greeting": text, "estimated_duration_ms": _estimate_tts_ms(text, "calm")})


@app.route("/send", methods=["POST"])
def send():
    """
    Thin wrapper around _send_impl() that guarantees the state machine can
    never get stuck: ANY uncaught exception anywhere in the real handler
    (including ones not individually audited below) is caught here, state
    is force-reset to LISTENING immediately (not left for the 45s watchdog
    in voice.py to eventually catch — this is the fast path, the watchdog
    is the backstop for anything that somehow still slips past this), and a
    safe fallback response is returned so the frontend's _busy flag clears
    and the input bar never goes permanently unresponsive.
    """
    try:
        return _send_impl()
    except Exception as e:
        print(f"[app_web] /send crashed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        voice.set_assistant_state("LISTENING")
        return jsonify({"response": "Something went quiet — try again.", "mood": "calm",
                        "confidence": "Low", "url": None})


def _send_impl():
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"response": "", "mood": "calm", "confidence": "Low", "url": None})
    voice.turn_log(f"/send request received by Flask: '{message}'")

    # Ground-truth self-report intercept (evaluation labelling): "label happy",
    # "that felt stressed", etc. Stores the label against the PREVIOUS turn's
    # mood reading and answers with a tiny ack — no LLM round trip, so it
    # never disturbs the natural conversation/barge-in flow.
    import re as _re
    _label = _re.match(
        r"^\s*(?:label|that felt|i felt)\s+(calm|stressed|happy|sad|excited|tired|anxious|frustrated|grateful|confused)\b",
        message.lower())
    if _label:
        gt = _label.group(1)
        row_id = database.set_ground_truth_mood(_user_id, gt)
        ack = (f"Noted — I've recorded that as '{gt}'. Thanks, it helps me learn."
               if row_id else "I don't have a turn to attach that to yet — talk to me first, then label it.")
        threading.Thread(target=voice.speak, args=(ack, "calm"), daemon=True).start()
        return jsonify({"response": ack, "mood": "calm", "confidence": "High", "url": None,
                        "estimated_duration_ms": _estimate_tts_ms(ack, "calm")})

    # C2: "Why do you think that?" — bypass the LLM entirely and answer with
    # the ACTUAL KNN neighbors (or honest rule-based explanation). No round-trip.
    if _user_id is not None and brain.detect_why_question(message):
        try:
            explanation = brain.build_why_explanation(
                brain._last_knn_neighbors,
                brain._last_prediction_was_rule_based,
            )
            threading.Thread(target=voice.speak, args=(explanation, "calm"), daemon=True).start()
            database.log_adaptation(
                _user_id, "explained_reasoning",
                f"Rule-based={brain._last_prediction_was_rule_based}, "
                f"neighbors={len(brain._last_knn_neighbors)}",
            )
            return jsonify({"response": explanation, "mood": "calm", "confidence": "High",
                            "url": None,
                            "estimated_duration_ms": _estimate_tts_ms(explanation, "calm")})
        except Exception as e:
            print(f"[app_web] C2 why-explanation failed: {e}")
            # Fall through to normal pipeline on any error.

    # C3: "How has this week been?" — generate the weekly digest locally
    # (no LLM call) and speak it. Purely descriptive, never prescriptive.
    if _user_id is not None and brain.detect_digest_request(message):
        try:
            digest = patterns.generate_weekly_digest(_user_id)
            digest_text = digest["digest_text"]
            threading.Thread(target=voice.speak, args=(digest_text, "calm"), daemon=True).start()
            database.log_adaptation(
                _user_id, "weekly_digest_viewed",
                f"Mood trend: {digest.get('mood_trend')}, "
                f"convos: {digest.get('conversations_this_week')}, "
                f"corrections: {digest.get('correction_count')}",
            )
            return jsonify({"response": digest_text, "mood": "calm", "confidence": "High",
                            "url": None,
                            "estimated_duration_ms": _estimate_tts_ms(digest_text, "calm")})
        except Exception as e:
            print(f"[app_web] C3 digest generation failed: {e}")
            # Fall through to normal pipeline on any error.


    fs       = _face_signals()

    # Fail-open True when unverifiable (no camera/no enrollment/check error) —
    # see face.FaceAnalyzer and _face_signals(). Gates writes to the
    # registered user's conversations/patterns/mood_readings tables below so
    # an unverified speaker's turn can't contaminate that user's learned
    # profile; personalization reads are gated the same way inside
    # brain.build_prompt (via run_parallel, which reads this same flag off `fs`).
    speaker_verified = fs.get("speaker_verified", True)
    # Sticky, confidence-gated: a single misdetected utterance can't flip
    # the conversation language (see voice.resolve_language).
    language = voice.resolve_language(message)
    # Real acoustics when this turn came in by voice (set by /poll); zeros when typed.
    voice_features = _consume_voice_features()
    
    # Push updated telemetry signals silently to fusion
    fusion.update_telemetry(**fs)
    fusion.update_telemetry(**voice_features)

    voice.set_assistant_state("THINKING")

    # C1: detect mood correction BEFORE escalation/offline checks so it's
    # captured even on error paths. Pull the recent history here once — shared
    # later for proactive follow-up context.
    recent_history = database.get_conversation_history(_user_id, limit=3)
    if speaker_verified:
        try:
            corrected_mood, original_mood = brain.detect_mood_correction(message, recent_history)
            if corrected_mood:
                database.set_ground_truth_mood(_user_id, corrected_mood, source="user_correction")
                database.log_adaptation(
                    _user_id, "mood_corrected",
                    f"ARIA read '{original_mood}' → user corrected to '{corrected_mood}'",
                )
                print(f"[app_web] C1 mood correction detected: '{original_mood}' → '{corrected_mood}'")
        except Exception as e:
            print(f"[app_web] C1 mood correction check failed: {e}")

    # C4: check if the current message resolves any pending concern BEFORE the
    # LLM pipeline, so we can mark it resolved and skip the follow-up this turn.
    if speaker_verified:
        try:
            pending_concerns = database.get_pending_concerns(_user_id)
            for concern in pending_concerns:
                if brain.detect_concern_resolved(message, concern.get("concern_text", "")):
                    database.mark_concern_resolved(concern["id"])
                    print(f"[app_web] C4 concern resolved naturally: {concern['concern_text']!r}")
        except Exception as e:
            print(f"[app_web] C4 concern resolution check failed: {e}")

    # Serious-distress escalation is checked BEFORE anything else — including
    # the offline branch — because this is the one behavior that must never
    # be skipped or deprioritised. Checked here (not deeper in the normal
    # response pipeline) so it can never accidentally run through the mood-
    # suggestion machinery at all, not even once.
    if brain.detect_serious_distress(message) or patterns.has_sustained_severe_low_mood(_user_id):
        return _handle_escalation(message, fs, speaker_verified, voice_features, language)

    if not network_state.is_online():
        return _handle_offline_send(message, fs, speaker_verified, voice_features)


    try:
        voice.turn_log("AI request sent to Groq (brain.run_parallel)")
        result     = brain.run_parallel(message, _user_id, voice_features, fs, language)
        voice.turn_log("AI text response received from Groq")
        ai_response = result["ai_response"]
        text_mood   = result.get("text_mood", {})
        text_mood_label = text_mood.get("mood", "calm") if isinstance(text_mood, dict) else "calm"
    except Exception as e:
        print(f"[app_web] run_parallel failed: {e}")
        # Immediate reset, not left for the outer wrapper/45s watchdog — this
        # is the single most likely real-world trigger (Groq rate limits,
        # network blips, malformed responses), so it gets the fast path.
        voice.set_assistant_state("LISTENING")
        return jsonify({"response": "Something went quiet — try again.", "mood": "calm", "confidence": "Low", "url": None})

    # System action and text extraction via the new autonomous pattern
    clean, executed_tools = brain.process_interaction(ai_response, voice.speak_safe)

    try:
        fused = fusion.fuse_moods(
            voice_features.get("mood", "calm"),
            fs.get("fused_mood", "calm"),
            text_mood_label,
            voice_features.get("pitch", 0.0),
            fs.get("fatigue", 0.0),
            fs.get("engagement", 0.5),
        )
        fused_mood      = fused.get("mood", "calm")
        fused_intensity = fused.get("intensity", 0.5)
    except Exception as e:
        print(f"[app_web] fuse_moods failed: {e}")
        fused_mood, fused_intensity = text_mood_label, 0.5

    # Asynchronous background logging to preserve 0-blocking UI response
    if speaker_verified:
        import memory
        threading.Thread(
            target=memory.log_interaction, 
            args=(_user_id, message, clean, fused_mood, language, voice_features, fs), 
            daemon=True
        ).start()
    else:
        print(f"[app_web] Speaker not verified (distance="
              f"{fs.get('verification_distance')}) — skipping DB writes")

    voice.turn_log("Response sent back to frontend "
                   "(caption now GATED on playback — pre-fix it appeared here)")
    return jsonify(_sanitize_for_json({
        "response":             clean,
        "actions_triggered":    executed_tools,
        "status":               "ok",
        "mood":                 fused_mood,
    }))


@app.route("/face_mood")
def face_mood():
    fs = _face_signals()
    result = _sanitize_for_json({**fs, "visual_mood": _to_visual_mood(fs.get("fused_mood", "calm"))})
    return jsonify(result)


@app.route("/poll")
def poll():
    """Return one pending voice-command message from the listen thread, or null.

    The queue carries {"text", "features"} dicts — the acoustic features
    (pitch, speed, voice mood) analysed in the listener thread are stashed
    here so the /send call that immediately follows can fuse them in.
    """
    return jsonify({"text": None})


@app.route("/complete_task", methods=["POST"])
def complete_task():
    data    = request.get_json(silent=True) or {}
    task_id = data.get("task_id")
    if task_id is not None:
        try:
            database.complete_task(int(task_id))
        except Exception as e:
            print(f"[app_web] complete_task failed: {e}")
    return jsonify({"ok": True})


@app.route("/memory")
def memory():
    """Everything ARIA has learned: profile summary, top patterns, the 7-day
    mood chart, pending tasks, and the adaptation ledger (when learned data
    actually changed behaviour)."""
    try:
        summary = patterns.build_user_profile_summary(_user_id)
        pending = database.get_pending_tasks(_user_id)
        total   = database.get_total_conversations(_user_id)
        user    = database.get_user(_user_id) or {}
        mood_days   = patterns.get_daily_mood_scores(_user_id, days=7)
        adaptations = database.get_adaptation_log(_user_id, limit=10)
        top_patterns = database.get_patterns(_user_id, limit=8)
        return jsonify(_sanitize_for_json({
            "summary":     summary,
            "tasks":       pending,
            "total":       total,
            "created_at":  user.get("created_at", ""),
            "mood_days":   [[d, round(s, 3)] for d, s in mood_days],
            "adaptations": adaptations,
            "patterns":    top_patterns,
        }))
    except Exception as e:
        print(f"[app_web] /memory failed: {e}")
        return jsonify({"summary": "Memory unavailable.", "tasks": [], "total": 0,
                        "created_at": "", "mood_days": [], "adaptations": [], "patterns": []})


@app.route("/system_status")
def system_status():
    """Small, always-current snapshot for the UI's status indicator: online/
    offline, AC/battery, and which TTS/STT models are actually active right
    now — so behavior changes (lighter models on battery, offline mode) are
    visibly explained instead of just felt as the app acting differently."""
    on_battery = power_state.is_on_battery()
    chatterbox_active = (voice.TTS_ENGINE == "chatterbox") and not on_battery
    return jsonify({
        "online": network_state.is_online(),
        "on_battery": on_battery,
        "tts_engine": "kokoro" if (voice.TTS_ENGINE == "chatterbox" and on_battery) else voice.TTS_ENGINE,
        "chatterbox_available_but_skipped": (voice.TTS_ENGINE == "chatterbox") and on_battery,
        "whisper_model": voice.WHISPER_MODEL,
    })


@app.route("/settings/wake", methods=["POST"])
def settings_wake():
    """Toggle: ON = only respond when the utterance contains "Aria";
    OFF (default) = respond to any clear speech."""
    global _wake_enabled
    data = request.get_json(silent=True) or {}
    _wake_enabled = bool(data.get("enabled", False))
    voice.set_wake_required(_wake_enabled)
    return jsonify({"ok": True})


@app.route("/settings/cam", methods=["POST"])
def settings_cam():
    global _cam_enabled
    data = request.get_json(silent=True) or {}
    _cam_enabled = bool(data.get("enabled", False))
    print(f"[app_web] camera awareness {'enabled' if _cam_enabled else 'disabled'}")
    if _face_analyzer:
        if _cam_enabled and not _face_analyzer.running:
            threading.Thread(target=_face_analyzer.start, daemon=True).start()
        elif not _cam_enabled and _face_analyzer.running:
            threading.Thread(target=_face_analyzer.stop, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/mic_level")
def mic_level():
    return jsonify(voice.get_mic_state())
