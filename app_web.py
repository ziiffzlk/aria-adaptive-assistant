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
import patterns
import system_actions
import voice

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
    return {
        "emotion": "neutral", "confidence": 0.0,
        "fatigue": 0.0, "engagement": 0.5,
        "fused_mood": "calm", "face_detected": False,
    }


def _estimate_tts_ms(text: str, mood: str) -> int:
    """Estimate speech duration in ms — drives the caption word-reveal pacing.
    Engine-aware: Chatterbox-Turbo speaks ~3.3 words/sec (measured from its
    generated samples 2026-07-03) and ignores mood pacing; Kokoro runs the
    mood-speed formula. Using Kokoro pacing for Turbo made captions lag the
    voice by ~40%."""
    words = max(1, len(text.split()))
    if voice.TTS_ENGINE == "chatterbox":
        return int((words / 3.3) * 1000)
    speed = voice.KOKORO_SPEED_MAP.get((mood or "").lower(), voice.KOKORO_SPEED_MAP["default"])
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
    profile and periodically retrain the personal KNN mood classifier."""
    try:
        patterns.refresh_behavioural_profile(user_id)
        total = database.get_total_conversations(user_id)
        patterns.maybe_train_knn(user_id, total)
    except Exception as e:
        print(f"[app_web] post-turn learning failed: {e}")


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
        # Hold the greeting until the primary TTS (Chatterbox-Turbo) finishes
        # loading — otherwise the very first thing ARIA says comes out in the
        # Kokoro fallback voice and her identity audibly switches mid-session.
        # 90s cap: Turbo loads in ~14s alone but ~50-60s in-app (it queues
        # behind TensorFlow/mediapipe imports) — a 45s cap was measured to
        # time out and greet in the wrong voice anyway (smoke run 2026-07-07).
        engine = voice._get_tts_engine()
        waited = 0.0
        while not engine.is_primary_ready and waited < 90.0:
            time.sleep(0.5)
            waited += 0.5
        if waited:
            print(f"[app_web] Held greeting {waited:.1f}s for primary TTS "
                  f"({'ready' if engine.is_primary_ready else 'timed out — using fallback voice'})")
        voice.speak(text, "calm")

    threading.Thread(target=_speak_greeting, daemon=True).start()
    return jsonify({"greeting": text, "estimated_duration_ms": _estimate_tts_ms(text, "calm")})


@app.route("/send", methods=["POST"])
def send():
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

    voice.set_assistant_state("THINKING")
    fs       = _face_signals()
    # Sticky, confidence-gated: a single misdetected utterance can't flip
    # the conversation language (see voice.resolve_language).
    language = voice.resolve_language(message)
    # Real acoustics when this turn came in by voice (set by /poll); zeros when typed.
    voice_features = _consume_voice_features()

    try:
        voice.turn_log("AI request sent to Groq (brain.run_parallel)")
        result     = brain.run_parallel(message, _user_id, voice_features, fs, language)
        voice.turn_log("AI text response received from Groq")
        ai_response = result["ai_response"]
        text_mood   = result.get("text_mood", {})
        text_mood_label = text_mood.get("mood", "calm") if isinstance(text_mood, dict) else "calm"
    except Exception as e:
        print(f"[app_web] run_parallel failed: {e}")
        return jsonify({"response": "Something went quiet — try again.", "mood": "calm", "confidence": "Low", "url": None})

    # Web search second pass: if the model asked for live information,
    # fetch it and generate the grounded final reply.
    try:
        ai_response, searched = brain.resolve_search_if_needed(ai_response, message, language)
    except Exception as e:
        print(f"[app_web] search resolution failed: {e}")
        searched = False

    # System action: execute the allowlisted desktop request, if any.
    action = brain.extract_action(ai_response)
    action_result = None
    if action and not system_actions.target_was_requested(action["verb"], action["target"], message):
        # The model emitted an action the user never asked for (usually a
        # "helpful" substitute while refusing an unavailable app) — drop it.
        print(f"[app_web] blocked unrequested action {action['verb']}|{action['target']}")
        action = None
    if action:
        action_result = system_actions.perform(action["verb"], action["target"])
        if action_result["ok"] and action["verb"] == "open_app":
            # App usage is behavioural data: "opens notepad most evenings"
            # is exactly the kind of pattern ARIA exists to notice.
            try:
                database.update_pattern(_user_id, "app_opened", action["target"])
            except Exception:
                pass

    try:
        fused = brain.fuse_moods(
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

    confidence = brain.extract_confidence(ai_response)
    clean      = brain.clean_response(ai_response)
    task       = brain.extract_task(ai_response)
    url        = brain.extract_url(ai_response)

    # YOUTUBE: personalised cheer-up suggestion — becomes a plain results URL
    # (no API key) and rides the EXACT same delivery path as URL: below.
    yt_query = brain.extract_youtube(ai_response)
    if yt_query and not url:
        from urllib.parse import quote_plus
        url = f"https://www.youtube.com/results?search_query={quote_plus(yt_query)}"
        try:
            interests = {p["pattern_value"].lower()
                         for p in database.get_patterns_by_type(_user_id, "frequent_topic", limit=5)}
            tied = next((i for i in interests if i in yt_query.lower()), None)
            database.log_adaptation(
                _user_id, "personalized_suggestion",
                (f"Offered YouTube '{yt_query}' tied to learned interest '{tied}'" if tied
                 else f"Offered YouTube '{yt_query}' (generic fallback — no matching learned interest)"))
        except Exception as e:
            print(f"[app_web] suggestion logging failed: {e}")

    # If an action was requested, make the spoken reply reflect what actually
    # happened — especially refusals, which the model can't know about.
    if action_result is not None and not action_result["ok"]:
        clean = (clean + " " + action_result["spoken"]).strip()
    elif action_result is not None and not clean:
        clean = action_result["spoken"]

    # Style feedback ("shorter", "tell me more") becomes a learned preference.
    try:
        style = patterns.detect_style_feedback(message)
        if style:
            database.update_pattern(_user_id, "style_pref", style)
            database.log_adaptation(_user_id, "style_feedback_noted",
                                    f"User asked for a more {style} style — remembered for future replies")
    except Exception as e:
        print(f"[app_web] style feedback failed: {e}")

    try:
        hour = datetime.now().hour
        database.save_conversation(
            _user_id, message, clean, fused_mood, confidence, language,
            voice_features.get("pitch", 0.0),
            voice_features.get("speaking_speed", 0.0),
            fs.get("emotion", "neutral"),
            fs.get("fatigue", 0.0),
            fs.get("engagement", 0.5),
        )
        database.save_mood_reading(
            _user_id, voice_features.get("mood", "calm"), fs.get("fused_mood", "calm"),
            text_mood_label, fused_mood, fused_intensity,
        )
        patterns.update_all_patterns(
            _user_id, message, fused_mood, hour,
            language, fs.get("emotion", "neutral"),
        )
        if task:
            database.save_task(_user_id, task.get("description", ""), task.get("due_date"))
    except Exception as e:
        print(f"[app_web] DB/patterns update failed: {e}")

    # The user asked to open a website: do it (the UI never consumed `url`).
    if url:
        try:
            import webbrowser
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        except Exception as e:
            print(f"[app_web] open url failed: {e}")

    # Sparse ground-truth check-in: every 7th turn, append a one-line ask so
    # evaluation labels accumulate without nagging or fighting barge-in flow.
    try:
        total_now = database.get_total_conversations(_user_id)
        if total_now and total_now % 7 == 0:
            clean += (" By the way — quick check-in: how did that actually feel? "
                      "You can just say 'label calm', 'label stressed', or however it felt.")
    except Exception:
        pass

    # Learning pass (profile refresh + periodic KNN retrain) off the request path.
    threading.Thread(target=_post_turn_learning, args=(_user_id,), daemon=True).start()

    threading.Thread(
        target=voice.speak, args=(clean, fused_mood), daemon=True
    ).start()

    voice.turn_log("Response sent back to frontend "
                   "(caption now GATED on playback — pre-fix it appeared here)")
    return jsonify(_sanitize_for_json({
        "response":             clean,
        "mood":                 fused_mood,
        "confidence":           confidence,
        "url":                  url,
        "searched":             searched,
        "estimated_duration_ms": _estimate_tts_ms(clean, fused_mood),
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
    global _pending_voice_features
    try:
        item = _voice_queue.get_nowait()
    except queue.Empty:
        return jsonify({"text": None})

    if isinstance(item, dict):
        with _pending_lock:
            _pending_voice_features = item.get("features")
        return jsonify({"text": item.get("text")})
    return jsonify({"text": item})  # plain-string fallback (legacy callers)


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
