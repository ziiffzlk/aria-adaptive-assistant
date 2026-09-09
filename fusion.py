import threading

# A thread-safe global store for background telemetry.
_telemetry_lock = threading.Lock()
current_telemetry = {
    "pitch": 0.0,
    "speaking_speed": 0.0,
    "fatigue": 0.0,
    "engagement": 0.5,
    "face_emotion": "neutral",
    "voice_mood": "calm",
    "text_mood": "calm",
    "fused_mood": "calm",
    "speaker_verified": True
}

NEGATIVE_TEXT_MOODS = {"sad", "angry", "frustrated", "stressed", "anxious", "disgusted"}

MOOD_DESCRIPTIONS = {
    "calm": "calm and composed",
    "happy": "positive and happy",
    "sad": "sad or down",
    "angry": "angry",
    "frustrated": "frustrated",
    "stressed": "stressed and tense",
    "tired": "fatigued and tired",
    "anxious": "anxious or nervous",
    "distracted": "distracted",
    "engaged": "highly engaged",
    "surprised": "surprised"
}

def update_telemetry(**kwargs):
    """Securely update the silent memory vector with new sensor metrics."""
    with _telemetry_lock:
        for k, v in kwargs.items():
            if k in current_telemetry:
                current_telemetry[k] = v

def get_telemetry() -> dict:
    """Return a copy of the current telemetry vector."""
    with _telemetry_lock:
        return dict(current_telemetry)

def evaluate_emotional_context(pitch, facial_state):
    threshold_high = 300
    if pitch > threshold_high and facial_state in ("stressed", "fearful", "fear", "angry"):
        return "BEHAVIORAL OVERRIDE: High tension detected. Enforce immediate, no-nonsense answers. No pleasantries."
    return "Standard dynamic adaptation."

def fuse_moods(voice_mood, face_mood, text_mood, voice_pitch, fatigue, engagement):
    """
    Combine voice (25%), face (40%) and text (35%) mood signals into one
    fused mood via weighted voting, with physiological overrides for
    cases the vote alone would miss.
    """
    votes = {}
    votes[face_mood] = votes.get(face_mood, 0.0) + 0.40
    votes[text_mood] = votes.get(text_mood, 0.0) + 0.35
    votes[voice_mood] = votes.get(voice_mood, 0.0) + 0.25

    fused_mood = max(votes, key=votes.get)
    intensity = round(votes[fused_mood], 2)

    if fatigue and fatigue > 0.7:
        fused_mood = "tired"
        intensity = round(max(intensity, fatigue), 2)
    elif face_mood == "angry" and text_mood in NEGATIVE_TEXT_MOODS:
        fused_mood = "frustrated"
        intensity = round(max(intensity, 0.75), 2)
    elif voice_pitch and voice_pitch > 300 and face_mood in ("fearful", "fear"):
        fused_mood = "anxious"
        intensity = round(max(intensity, 0.75), 2)

    description = f"[INTERNAL SENSOR STATE: {fused_mood.upper()} - DO NOT READ ALOUD]"
    context_modifier = evaluate_emotional_context(voice_pitch, face_mood)
    update_telemetry(fused_mood=fused_mood)
    return {"mood": fused_mood, "intensity": intensity, "description": description, "context_modifier": context_modifier}
