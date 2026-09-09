import database
import patterns
from datetime import datetime

def get_memory_context(user_id: int) -> str:
    """
    Retrieves facts, preferences, and recent conversational history from the DB
    and formats them into the unified MEMORY CONTEXT block.
    """
    if not user_id:
        return ""

    sections = ["=== PERSISTENT MEMORY CONTEXT ==="]
    
    # 1. User Profile & Facts (Top 5 patterns)
    sections.append("User Profile & Facts:")
    try:
        profile = database.get_behavioural_profile(user_id)
        if profile and profile.get("style_pref"):
            sections.append(f"- Preferred conversational style: {profile['style_pref']}")
            
        user_patterns = database.get_patterns(user_id, limit=5)
        for p in user_patterns:
            ptype = p['pattern_type'].replace("_", " ")
            sections.append(f"- {ptype}: {p['pattern_value']}")
    except Exception as e:
        print(f"[memory] Failed to fetch facts: {e}")

    # 2. Recent Interaction History (Last 4 turns)
    sections.append("\nRecent Interaction History:")
    try:
        history = database.get_conversation_history(user_id, limit=4)
        if history:
            # history is returned newest first, so we reverse it to chronological order
            for turn in reversed(history):
                sections.append(f"- User: {turn['user_message']}")
                sections.append(f"- ARIA: {turn['ai_response']}")
        else:
            sections.append("- (No recent history)")
    except Exception as e:
        print(f"[memory] Failed to fetch history: {e}")

    return "\n".join(sections)


def log_interaction(user_id: int, message: str, clean_text: str, fused_mood: str, language: str, voice_features: dict, fs: dict):
    """
    Asynchronously executes all database writes (conversation log, patterns update, mood logging)
    and the post-turn learning pass.
    """
    if not user_id:
        return

    try:
        style = patterns.detect_style_feedback(message)
        if style:
            database.update_pattern(user_id, "style_pref", style)
            database.log_adaptation(user_id, "style_feedback_noted", f"User asked for a more {style} style")
    except Exception as e:
        print(f"[memory] style feedback failed: {e}")

    try:
        hour = datetime.now().hour
        # Log the core conversation turn
        database.save_conversation(
            user_id, message, clean_text, fused_mood, "Medium", language,
            voice_features.get("pitch", 0.0) if voice_features else 0.0,
            voice_features.get("speaking_speed", 0.0) if voice_features else 0.0,
            fs.get("emotion", "neutral") if fs else "neutral",
            fs.get("fatigue", 0.0) if fs else 0.0,
            fs.get("engagement", 0.5) if fs else 0.5,
        )
        
        # Log mood trajectory
        database.save_mood_reading(
            user_id, 
            voice_features.get("mood", "calm") if voice_features else "calm", 
            fs.get("fused_mood", "calm") if fs else "calm",
            "calm", # text_mood is obsolete/handled by fused_mood
            fused_mood, 
            0.5, # intensity
        )
        
        # Update dynamic patterns (KNN)
        patterns.update_all_patterns(
            user_id, message, fused_mood, hour, language, 
            fs.get("emotion", "neutral") if fs else "neutral"
        )
    except Exception as e:
        print(f"[memory] DB logging failed: {e}")

    # C4a: Detect new mentioned concerns
    try:
        import brain
        concern_text, window_hours = brain.detect_mentioned_concern(message)
        if concern_text:
            database.save_mentioned_concern(user_id, concern_text, window_hours)
            print(f"[memory] Concern saved: {concern_text!r}")
    except Exception as e:
        print(f"[memory] Concern save failed: {e}")

    # KNN offline retrain / profile refresh (formerly _post_turn_learning in app_web)
    try:
        import app_web
        if hasattr(app_web, '_post_turn_learning'):
            app_web._post_turn_learning(user_id)
    except Exception as e:
        print(f"[memory] Learning pass failed: {e}")
