"""
ARIA Desktop - Patterns Module
Behavioural pattern tracking, mood trend analysis, proactive suggestions,
user profile summarisation, and a per-session KNN mood classifier trained
on the user's own historical multimodal data.
"""

from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier

import database


TOPIC_KEYWORDS = [
    "help", "work", "study", "code", "deadline", "task", "tired", "stressed",
    "happy", "project", "meeting", "family", "health", "food", "exercise", "money",
]

# Rough positivity weighting per mood label, used for trend/day analysis.
MOOD_POSITIVITY = {
    "happy": 1.0, "excited": 0.9, "calm confident": 0.8, "calm": 0.6, "engaged": 0.5,
    "surprised": 0.3, "distracted": -0.1, "tired": -0.4, "anxious": -0.6,
    "sad": -0.7, "frustrated": -0.7, "stressed": -0.8,
}

# A single, "currently active" KNN model - this is a single-user desktop
# session (one name prompt at startup), so one module-level model is enough.
_active_model = None
_active_user_id = None
# C2: the ordered list of training rows that built the current model; indices
# returned by kneighbors() map directly back into this list so explain
# functions can retrieve the actual historical samples (date, mood, features).
_active_training_rows: list = []
# C2: True when the LAST mood prediction used the rule-based fallback (no KNN
# model trained yet), False when KNN drove it. Used by build_why_explanation.
_last_used_rule_based: bool = True


# ----------------------------------------------------------------------
# Pattern tracking
# ----------------------------------------------------------------------

def update_all_patterns(user_id, message, mood, hour, language, face_emotion):
    """Update active_hour, frequent_topic, dominant_mood, language_used and face_emotion_pattern."""
    if message:
        lowered = message.lower()
        for keyword in TOPIC_KEYWORDS:
            if keyword in lowered:
                database.update_pattern(user_id, "frequent_topic", keyword)

    if hour is not None:
        database.update_pattern(user_id, "active_hour", str(hour))

    if mood:
        database.update_pattern(user_id, "dominant_mood", mood)

    if language:
        database.update_pattern(user_id, "language_used", language)

    if face_emotion:
        database.update_pattern(user_id, "face_emotion_pattern", face_emotion)


# Phrases that signal the user wants a different response style — the raw
# material for the learned style preference fed back into brain.build_prompt.
_CONCISE_MARKERS = [
    "shorter", "too long", "be brief", "keep it short", "less detail",
    "get to the point", "too much", "stop rambling", "briefly",
]
_DETAILED_MARKERS = [
    "tell me more", "more detail", "explain more", "go deeper", "elaborate",
    "expand on", "in depth", "longer answer", "full explanation",
]


def detect_style_feedback(message):
    """Return 'concise', 'detailed', or None based on explicit style feedback in a message."""
    lowered = (message or "").lower()
    if any(m in lowered for m in _CONCISE_MARKERS):
        return "concise"
    if any(m in lowered for m in _DETAILED_MARKERS):
        return "detailed"
    return None


def refresh_behavioural_profile(user_id):
    """
    Recompute the cached behavioural profile from raw patterns/mood history and
    persist it. Called after every conversational turn so brain.build_prompt and
    the Memory panel always read a current picture without recomputing it.
    """
    try:
        trend = detect_mood_trend(user_id)
        top_hour  = database.get_patterns_by_type(user_id, "active_hour", limit=1)
        top_topic = database.get_patterns_by_type(user_id, "frequent_topic", limit=1)
        top_lang  = database.get_patterns_by_type(user_id, "language_used", limit=1)
        top_face  = database.get_patterns_by_type(user_id, "face_emotion_pattern", limit=1)

        database.upsert_behavioural_profile(
            user_id,
            dominant_mood=trend.get("dominant_mood"),
            mood_trend=trend.get("trend"),
            active_hour=top_hour[0]["pattern_value"] if top_hour else None,
            frequent_topic=top_topic[0]["pattern_value"] if top_topic else None,
            language_used=top_lang[0]["pattern_value"] if top_lang else None,
            face_emotion_pattern=top_face[0]["pattern_value"] if top_face else None,
            summary=build_user_profile_summary(user_id),
        )
    except Exception as e:
        print(f"[patterns] refresh_behavioural_profile failed: {e}")


def maybe_train_knn(user_id, total_conversations):
    """
    Retrain the personal KNN mood classifier every 5 turns (and on startup via
    total==None). Logs to the adaptation ledger when training succeeds so the
    UI can show the model growing with the user.
    """
    if total_conversations is not None and total_conversations % 5 != 0:
        return
    try:
        if train_knn(user_id):
            rows = database.get_all_conversations(user_id)
            usable = sum(1 for r in rows if r.get("voice_pitch") not in (None, 0.0))
            database.log_adaptation(
                user_id, "knn_retrained",
                f"Personal mood classifier retrained on {usable} voice-feature samples",
            )
    except Exception as e:
        print(f"[patterns] maybe_train_knn failed: {e}")


# ----------------------------------------------------------------------
# Mood trend analysis
# ----------------------------------------------------------------------

def detect_mood_trend(user_id):
    """
    Analyse mood_readings from the last 7 days.
    Returns dict: {dominant_mood, trend, most_stressed_day, calmest_day}
    trend is one of 'improving', 'declining', 'stable', or 'insufficient_data'.
    """
    readings = database.get_mood_history(user_id, days=7)
    if not readings:
        return {"dominant_mood": None, "trend": "insufficient_data",
                "most_stressed_day": None, "calmest_day": None}

    mood_counts = {}
    day_scores = {}

    for r in readings:
        mood = r.get("fused_mood") or "calm"
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

        intensity = r.get("intensity") if r.get("intensity") is not None else 0.5
        score = MOOD_POSITIVITY.get(mood, 0.0) * intensity

        timestamp = r.get("timestamp") or ""
        day = timestamp.split(" ")[0].split("T")[0] if timestamp else "unknown"
        day_scores.setdefault(day, []).append(score)

    dominant_mood = max(mood_counts, key=mood_counts.get)
    avg_by_day = {day: sum(scores) / len(scores) for day, scores in day_scores.items()}
    most_stressed_day = min(avg_by_day, key=avg_by_day.get)
    calmest_day = max(avg_by_day, key=avg_by_day.get)

    sorted_days = sorted(avg_by_day.keys())
    if len(sorted_days) >= 2:
        midpoint = max(len(sorted_days) // 2, 1)
        first_half = sorted_days[:midpoint]
        second_half = sorted_days[midpoint:] or sorted_days[-1:]
        first_avg = sum(avg_by_day[d] for d in first_half) / len(first_half)
        second_avg = sum(avg_by_day[d] for d in second_half) / len(second_half)
        diff = second_avg - first_avg
        trend = "improving" if diff > 0.15 else "declining" if diff < -0.15 else "stable"
    else:
        trend = "stable"

    return {
        "dominant_mood": dominant_mood,
        "trend": trend,
        "most_stressed_day": most_stressed_day,
        "calmest_day": calmest_day,
    }


def get_daily_mood_scores(user_id, days=7):
    """
    Per-day average mood positivity score for the last `days` days, oldest
    first - used for charting (eg. the Memory dashboard's 7-day bar chart).
    Returns a list of (day_str, score) tuples; score is roughly -1..1.
    """
    readings = database.get_mood_history(user_id, days=days)
    day_scores = {}
    for r in readings:
        mood = r.get("fused_mood") or "calm"
        intensity = r.get("intensity") if r.get("intensity") is not None else 0.5
        score = MOOD_POSITIVITY.get(mood, 0.0) * intensity
        timestamp = r.get("timestamp") or ""
        day = timestamp.split(" ")[0].split("T")[0] if timestamp else "unknown"
        day_scores.setdefault(day, []).append(score)
    return [(day, sum(scores) / len(scores)) for day, scores in sorted(day_scores.items())]


def is_mood_sustained(user_id, mood, min_turns=3):
    """
    True if the last `min_turns` real mood_readings all share the same
    fused-mood category as `mood` — distinguishes "this has come up
    consistently" from a single momentary reading. Used to gate Part B1's
    concrete-action offers: a mood glimpsed once should never trigger a
    suggestion, only a mood that's genuinely persisted across several turns.
    """
    recent = database.get_recent_fused_moods(user_id, min_turns)
    if len(recent) < min_turns:
        return False
    return all((m or "").lower() == (mood or "").lower() for m in recent)


def has_sustained_severe_low_mood(user_id, days=14, min_negative_ratio=0.7, min_readings=6):
    """
    True if fused mood has been persistently severe-negative across MANY
    real sessions/days — distinct from "stressed about today's deadline".
    Deliberately narrower than all negative moods (excludes ordinary
    stressed/frustrated/anxious) and requires both a real minimum sample
    count and a real time span, so a single bad afternoon can't trigger it.
    Used alongside brain.detect_serious_distress (explicit language in the
    CURRENT message) as the other half of the Part B3 escalation trigger —
    either a sustained pattern OR explicit crisis language should escalate.
    """
    readings = database.get_mood_history(user_id, days=days)
    if len(readings) < min_readings:
        return False
    severe = {"sad", "hopeless", "despair"}
    negative_count = sum(1 for r in readings if (r.get("fused_mood") or "").lower() in severe)
    return (negative_count / len(readings)) >= min_negative_ratio


# ----------------------------------------------------------------------
# Proactive suggestions
# ----------------------------------------------------------------------

# How long a given suggestion (same task, same topic nudge) stays suppressed
# after being surfaced once. Without this, get_proactive_suggestion is a
# pure function of (pending_tasks, patterns, hour) with no memory of having
# just said it — it returned the literal same sentence on every single call
# (confirmed live: 5/5 identical). "Until the task changes/completes" is
# already handled for free — a completed task drops out of pending_tasks —
# this cooldown covers the "still pending, but don't repeat it every turn" case.
PROACTIVE_SUGGESTION_COOLDOWN_MINUTES = 30


def _suggestion_on_cooldown(user_id, suggestion_key):
    from datetime import datetime as _dt, timedelta as _td
    now = _dt.now()

    # Global 5-minute cooldown between ANY proactive messages
    last_global = database.get_suggestion_last_surfaced(user_id, "__GLOBAL_PROACTIVE__")
    if last_global:
        try:
            if now - _dt.fromisoformat(last_global) < _td(minutes=5):
                return True
        except ValueError:
            pass

    # Specific 30-minute cooldown for the same suggestion
    last = database.get_suggestion_last_surfaced(user_id, suggestion_key)
    if not last:
        return False
    try:
        elapsed = now - _dt.fromisoformat(last)
    except ValueError:
        return False
    return elapsed < _td(minutes=PROACTIVE_SUGGESTION_COOLDOWN_MINUTES)


def get_proactive_suggestion(user_id, current_hour, pending_tasks, patterns):
    """
    Suggest something relevant based on pending tasks or recurring patterns.
    Priority: pending task reminder > time+topic correlation > general frequent topic.
    Returns (suggestion_text, suggestion_key), or (None, None) if there's
    nothing to say OR the best candidate was already surfaced within the
    cooldown window. Caller (brain.build_prompt) is responsible for calling
    database.mark_suggestion_surfaced(user_id, suggestion_key) if it actually
    uses the text — this function only reads, it doesn't start the cooldown
    itself (a candidate that gets computed but then isn't shown for some
    other reason shouldn't be marked as shown).
    """
    if pending_tasks:
        task = pending_tasks[0]
        key = f"task:{task.get('id')}"
        if _suggestion_on_cooldown(user_id, key):
            return None, None
        desc = task.get("task_description")
        due = task.get("due_date")
        if due:
            return f"By the way, don't forget: \"{desc}\" is due {due}.", key
        return f"By the way, you still have a pending task: \"{desc}\".", key

    if not patterns:
        return None, None

    active_hours = [p for p in patterns if p.get("pattern_type") == "active_hour"]
    topics = [p for p in patterns if p.get("pattern_type") == "frequent_topic"]

    hour_match = next(
        (p for p in active_hours if str(p.get("pattern_value")) == str(current_hour) and p.get("frequency", 0) >= 3),
        None,
    )
    if hour_match and topics:
        top_topic = max(topics, key=lambda p: p.get("frequency", 0))
        key = f"hour_topic:{current_hour}:{top_topic['pattern_value']}"
        if _suggestion_on_cooldown(user_id, key):
            return None, None
        return (f"You're often active around this time and tend to bring up "
                f"{top_topic['pattern_value']} - want to pick that up again?", key)

    if topics:
        top_topic = max(topics, key=lambda p: p.get("frequency", 0))
        if top_topic.get("frequency", 0) >= 3:
            key = f"topic:{top_topic['pattern_value']}"
            if _suggestion_on_cooldown(user_id, key):
                return None, None
            return (f"You've mentioned {top_topic['pattern_value']} a few times recently - "
                    f"how's that going?", key)

    return None, None


# ----------------------------------------------------------------------
# User profile summary
# ----------------------------------------------------------------------

def build_user_profile_summary(user_id):
    """Summarise everything learned about a user, for use in prompts or the Memory tab."""
    user = database.get_user(user_id)
    if not user:
        return "No profile data available yet."

    patterns = database.get_patterns(user_id, limit=10)
    total_conversations = database.get_total_conversations(user_id)
    pending_tasks = database.get_pending_tasks(user_id)
    trend = detect_mood_trend(user_id)

    lines = [
        f"Name: {user['name']}",
        f"Total conversations: {total_conversations}",
        f"Personality mode: {user.get('personality_mode', 'friendly')}",
        f"Preferred language: {user.get('language_preference', 'en')}",
    ]

    topic_patterns = [p for p in patterns if p["pattern_type"] == "frequent_topic"]
    if topic_patterns:
        lines.append("Frequent topics: " + ", ".join(p["pattern_value"] for p in topic_patterns[:5]))

    hour_patterns = [p for p in patterns if p["pattern_type"] == "active_hour"]
    if hour_patterns:
        lines.append("Most active hours: " + ", ".join(f"{p['pattern_value']}:00" for p in hour_patterns[:3]))

    if trend.get("dominant_mood"):
        lines.append(f"Dominant mood lately: {trend['dominant_mood']} (trend: {trend['trend']})")

    if pending_tasks:
        lines.append(f"Pending tasks: {len(pending_tasks)}")

    return "\n".join(lines)


def evaluate_cognitive_state(user_id, message, context):
    """
    Evaluates interaction velocity and task complexity to return a cognitive state:
    'FAST-PACED / EXECUTION STATE', 'DEEP-WORK / ANALYTICAL STATE', or 'AMBIGUOUS / SHORT INPUTS'.
    """
    if not message:
        return "AMBIGUOUS / SHORT INPUTS"
    
    lowered = message.lower()
    words = lowered.split()
    word_count = len(words)
    
    # Task complexity indicators
    complex_markers = ["how to", "explain", "design", "why", "analyze", "debug", "architect", "compare", "build"]
    is_complex = any(m in lowered for m in complex_markers)
    
    if is_complex or word_count >= 15:
        return "DEEP-WORK / ANALYTICAL STATE"
    elif word_count <= 4:
        # Check if it's a direct command
        exec_markers = ["run", "do", "open", "close", "start", "stop", "set", "play", "pause", "remind", "test"]
        if any(m in lowered for m in exec_markers):
            return "FAST-PACED / EXECUTION STATE"
        return "AMBIGUOUS / SHORT INPUTS"
    else:
        return "FAST-PACED / EXECUTION STATE"


# ----------------------------------------------------------------------
# KNN mood classifier
# ----------------------------------------------------------------------

def train_knn(user_id):
    """
    Train a k=3 KNN mood classifier on this user's own historical multimodal
    data (voice_pitch, voice_speed, fatigue_level, engagement_level -> mood),
    pulled from the conversations table (the mood_readings table tracks mood
    labels only, not the raw acoustic/visual features needed as inputs here).
    Only trains if more than 10 complete samples exist. Returns True/False.
    """
    global _active_model, _active_user_id, _active_training_rows

    rows = database.get_all_conversations(user_id)
    samples, labels, usable_rows = [], [], []
    for r in rows:
        if (r.get("voice_pitch") is None or r.get("voice_speed") is None or
                r.get("fatigue_level") is None or r.get("engagement_level") is None or not r.get("mood")):
            continue
        # pitch == 0.0 means the turn was typed (no acoustic data captured) —
        # those rows would teach the classifier that silence means "calm".
        if r["voice_pitch"] == 0.0:
            continue
        samples.append([r["voice_pitch"], r["voice_speed"], r["fatigue_level"], r["engagement_level"]])
        labels.append(r["mood"])
        usable_rows.append(r)  # C2: keep parallel to samples for neighbor lookup

    if len(samples) <= 10:
        print(f"[patterns] Not enough complete multimodal samples to train KNN ({len(samples)}/11 minimum)")
        return False

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(samples, labels)
    _active_model = model
    _active_user_id = user_id
    _active_training_rows = usable_rows  # C2: store for neighbor lookup
    print(f"[patterns] KNN mood classifier trained for user {user_id} on {len(samples)} samples")
    return True


def predict_mood_knn(features_dict):
    """
    Predict a mood label from the trained KNN model given a features dict
    with voice_pitch (or pitch), voice_speed (or speaking_speed), fatigue,
    and engagement. Returns the predicted mood string, or None if untrained.
    """
    if _active_model is None:
        return None
    try:
        x = [[
            features_dict.get("voice_pitch", features_dict.get("pitch", 0.0)),
            features_dict.get("voice_speed", features_dict.get("speaking_speed", 0.0)),
            features_dict.get("fatigue", 0.0),
            features_dict.get("engagement", 0.5),
        ]]
        return _active_model.predict(x)[0]
    except Exception as e:
        print(f"[patterns] predict_mood_knn failed: {e}")
        return None


def predict_mood_knn_with_neighbors(features_dict):
    """
    C2: Like predict_mood_knn(), but also returns the k=3 actual historical
    training rows that drove the prediction — enabling honest explainability.

    Returns (predicted_mood: str | None, neighbors: list[dict]).
    neighbors is a list of up to 3 dicts from the original conversation rows
    (keys: timestamp, mood, voice_pitch, voice_speed). Empty list when KNN
    is untrained; callers should check for None mood to detect rule-based path.
    """
    global _last_used_rule_based
    if _active_model is None:
        _last_used_rule_based = True
        return None, []
    try:
        x = [[
            features_dict.get("voice_pitch", features_dict.get("pitch", 0.0)),
            features_dict.get("voice_speed", features_dict.get("speaking_speed", 0.0)),
            features_dict.get("fatigue", 0.0),
            features_dict.get("engagement", 0.5),
        ]]
        mood = _active_model.predict(x)[0]
        # kneighbors returns (distances, indices) — indices index into the
        # training set IN THE ORDER IT WAS FIT, which is _active_training_rows.
        distances, indices = _active_model.kneighbors(x, n_neighbors=min(3, len(_active_training_rows)))
        neighbors = []
        for idx in indices[0]:
            if 0 <= idx < len(_active_training_rows):
                r = _active_training_rows[idx]
                neighbors.append({
                    "timestamp": r.get("timestamp", ""),
                    "mood":      r.get("mood", ""),
                    "pitch":     r.get("voice_pitch", 0.0),
                    "speed":     r.get("voice_speed", 0.0),
                })
        _last_used_rule_based = False
        return mood, neighbors
    except Exception as e:
        print(f"[patterns] predict_mood_knn_with_neighbors failed: {e}")
        _last_used_rule_based = True
        return None, []


# ----------------------------------------------------------------------
# C3 — Weekly reflection digest
# ----------------------------------------------------------------------

def generate_weekly_digest(user_id):
    """
    Synthesise a purely descriptive weekly reflection from real accumulated
    data. Never prescriptive (no 'you should' language). Called only when
    the user explicitly asks — never pushed uninvited.

    Returns a dict with structured data fields PLUS a pre-built digest_text
    string ready to speak aloud. All analysis is done locally (no LLM call)
    so the output is deterministic and can't hallucinate.
    """
    # — This week vs previous week —
    this_week_readings  = database.get_mood_history(user_id, days=7)
    all_two_weeks       = database.get_mood_history(user_id, days=14)
    prev_week_readings  = [r for r in all_two_weeks if r not in this_week_readings]

    def _dominant(readings):
        counts = {}
        for r in readings:
            m = r.get("fused_mood") or "calm"
            counts[m] = counts.get(m, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def _avg_positivity(readings):
        if not readings:
            return None
        scores = [MOOD_POSITIVITY.get(r.get("fused_mood") or "calm", 0.0) * (r.get("intensity") or 0.5)
                  for r in readings]
        return round(sum(scores) / len(scores), 3)

    trend_this  = detect_mood_trend(user_id)  # uses last 7 days
    dominant_this  = _dominant(this_week_readings)
    dominant_prev  = _dominant(prev_week_readings)
    avg_this  = _avg_positivity(this_week_readings)
    avg_prev  = _avg_positivity(prev_week_readings)

    # — Active hour —
    top_hour_this = database.get_patterns_by_type(user_id, "active_hour", limit=1)
    active_hour   = top_hour_this[0]["pattern_value"] if top_hour_this else None

    # — Frequent topics this week —
    topic_rows  = database.get_patterns_by_type(user_id, "frequent_topic", limit=5)
    top_topics  = [p["pattern_value"] for p in topic_rows]

    # — Conversation count this week —
    from datetime import timedelta
    since_7d = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
    all_convs = database.get_all_conversations(user_id)
    convs_this_week = sum(
        1 for c in all_convs
        if (c.get("timestamp") or "") >= since_7d
    )

    # — Mood correction count (C1 improvement proxy) —
    correction_count = database.get_recent_concern_count(user_id, "mood_corrected", days=7)

    # — Build spoken digest text —
    lines = []

    # Mood overview
    if dominant_this:
        lines.append(f"Over the past week, {dominant_this} has been the most common mood.")
    if trend_this.get("trend") and trend_this["trend"] != "insufficient_data":
        trend_label = trend_this["trend"]
        if trend_label == "improving":
            lines.append("Overall, things looked a bit more positive toward the end of the week.")
        elif trend_label == "declining":
            lines.append("The mood signal trended a little lower toward the end of the week.")
        else:
            lines.append("Mood has been fairly consistent throughout the week.")

    # Week-over-week comparison
    if avg_this is not None and avg_prev is not None:
        diff = avg_this - avg_prev
        if diff > 0.05:
            lines.append("Compared to the week before, things felt a bit lighter this week.")
        elif diff < -0.05:
            lines.append("Compared to the week before, this week had a slightly heavier tone.")
        elif dominant_prev:
            lines.append(f"The mood pattern was fairly similar to the week before, when {dominant_prev} was most common.")

    # Conversations
    if convs_this_week:
        lines.append(f"You had {convs_this_week} conversation{'s' if convs_this_week != 1 else ''} with me this week.")

    # Active hour
    if active_hour is not None:
        try:
            h = int(active_hour)
            hour_str = f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"
            lines.append(f"You've most often been active around {hour_str}.")
        except (ValueError, TypeError):
            pass

    # Topics
    if top_topics:
        topic_list = ", ".join(top_topics[:3])
        lines.append(f"Topics that came up most often included {topic_list}.")

    # Correction signal
    if correction_count > 0:
        lines.append(
            f"You corrected my mood reading {correction_count} time{'s' if correction_count != 1 else ''} — "
            "that kind of feedback helps me calibrate better over time."
        )

    if not lines:
        digest_text = ("I don't have enough data from this week to put together a reflection yet. "
                       "Keep talking to me and I'll have more to share.")
    else:
        digest_text = " ".join(lines)

    return {
        "dominant_mood_this_week":  dominant_this,
        "dominant_mood_prev_week":  dominant_prev,
        "mood_trend":               trend_this.get("trend"),
        "avg_positivity_this_week": avg_this,
        "avg_positivity_prev_week": avg_prev,
        "active_hour":              active_hour,
        "top_topics":               top_topics,
        "conversations_this_week":  convs_this_week,
        "correction_count":         correction_count,
        "digest_text":              digest_text,
    }


# ----------------------------------------------------------------------
# C5 — User baseline speaking rate
# ----------------------------------------------------------------------

def get_user_baseline_speaking_rate(user_id, min_samples=10):
    """
    Compute the user's median speaking rate (words/sec) from their real
    accumulated voice conversation history.

    Uses the median of up to the last 50 voice turns (voice_speed > 0.0;
    typed turns have voice_speed == 0.0 and are excluded). Returns None when
    fewer than min_samples voice turns exist — same minimum-data principle
    as the KNN classifier's 11-sample floor: don't adapt until there's enough
    real signal to be meaningful.
    """
    rows = database.get_all_conversations(user_id)
    # most-recent-first so we take the last 50 voice turns efficiently
    voice_speeds = [
        r["voice_speed"]
        for r in reversed(rows)
        if (r.get("voice_speed") or 0.0) > 0.0
    ][:50]

    if len(voice_speeds) < min_samples:
        print(f"[patterns] Not enough voice samples for pace calibration "
              f"({len(voice_speeds)}/{min_samples} minimum)")
        return None

    import statistics
    baseline = round(statistics.median(voice_speeds), 3)
    print(f"[patterns] User baseline speaking rate: {baseline:.3f} wps "
          f"(median of {len(voice_speeds)} voice turns)")
    return baseline


if __name__ == "__main__":
    # Quick manual smoke test: python patterns.py
    database.init_db()
    uid = database.get_or_create_user("Pattern Test User")

    for hour, msg, mood, lang, face in [
        (9, "I need help with my project deadline", "stressed", "en", "neutral"),
        (9, "Work is stressing me out, the project deadline is close", "stressed", "en", "sad"),
        (14, "Just had a great workout, feeling happy", "happy", "en", "happy"),
        (9, "Another deadline for the project, I'm tired", "tired", "en", "tired"),
    ]:
        update_all_patterns(uid, msg, mood, hour, lang, face)

    print("Patterns:", database.get_patterns(uid))
    print("Profile summary:\n", build_user_profile_summary(uid))

    database.save_task(uid, "Submit project", "tomorrow")
    tasks = database.get_pending_tasks(uid)
    patterns = database.get_patterns(uid)
    print("Suggestion:", get_proactive_suggestion(uid, 9, tasks, patterns))
    print("Suggestion (no tasks):", get_proactive_suggestion(uid, 9, [], patterns))

    for mood in ["happy", "stressed", "calm", "stressed", "happy", "tired", "calm", "stressed", "happy", "calm", "tired", "stressed"]:
        database.save_mood_reading(uid, mood, mood, mood, mood, 0.7)
    print("Mood trend:", detect_mood_trend(uid))

    import random
    for i in range(15):
        database.save_conversation(
            uid, f"test message {i}", f"test response {i}",
            random.choice(["happy", "stressed", "calm", "tired"]), "High", "en",
            random.uniform(100, 250), random.uniform(1, 4), "neutral",
            random.uniform(0, 1), random.uniform(0, 1),
        )
    trained = train_knn(uid)
    print("KNN trained:", trained)
    if trained:
        print("Predicted mood:", predict_mood_knn({"pitch": 220, "speaking_speed": 3.5, "fatigue": 0.1, "engagement": 0.8}))
