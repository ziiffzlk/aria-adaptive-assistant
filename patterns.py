"""
ARIA Desktop - Patterns Module
Behavioural pattern tracking, mood trend analysis, proactive suggestions,
user profile summarisation, and a per-session KNN mood classifier trained
on the user's own historical multimodal data.
"""

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


# ----------------------------------------------------------------------
# Proactive suggestions
# ----------------------------------------------------------------------

def get_proactive_suggestion(user_id, current_hour, pending_tasks, patterns):
    """
    Suggest something relevant based on pending tasks or recurring patterns.
    Priority: pending task reminder > time+topic correlation > general frequent topic.
    Returns a suggestion string, or None.
    """
    if pending_tasks:
        task = pending_tasks[0]
        desc = task.get("task_description")
        due = task.get("due_date")
        if due:
            return f"By the way, don't forget: \"{desc}\" is due {due}."
        return f"By the way, you still have a pending task: \"{desc}\"."

    if not patterns:
        return None

    active_hours = [p for p in patterns if p.get("pattern_type") == "active_hour"]
    topics = [p for p in patterns if p.get("pattern_type") == "frequent_topic"]

    hour_match = next(
        (p for p in active_hours if str(p.get("pattern_value")) == str(current_hour) and p.get("frequency", 0) >= 3),
        None,
    )
    if hour_match and topics:
        top_topic = max(topics, key=lambda p: p.get("frequency", 0))
        return f"You're often active around this time and tend to bring up {top_topic['pattern_value']} - want to pick that up again?"

    if topics:
        top_topic = max(topics, key=lambda p: p.get("frequency", 0))
        if top_topic.get("frequency", 0) >= 3:
            return f"You've mentioned {top_topic['pattern_value']} a few times recently - how's that going?"

    return None


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
        lines.append(f"Recent dominant mood: {trend['dominant_mood']} (trend: {trend['trend']})")

    if pending_tasks:
        lines.append(f"Pending tasks: {len(pending_tasks)}")

    return "\n".join(lines)


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
    global _active_model, _active_user_id

    rows = database.get_all_conversations(user_id)
    samples, labels = [], []
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

    if len(samples) <= 10:
        print(f"[patterns] Not enough complete multimodal samples to train KNN ({len(samples)}/11 minimum)")
        return False

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(samples, labels)
    _active_model = model
    _active_user_id = user_id
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
