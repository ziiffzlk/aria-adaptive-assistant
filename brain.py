"""
ARIA Desktop - Brain Module
LLM reasoning via Groq: multimodal mood fusion, prompt construction,
response generation, personalised greetings, and response parsing
(confidence / task / URL extraction).
"""

import os
import re
import time
import concurrent.futures
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq, RateLimitError

import database
import system_actions
# Aliased: build_prompt/run_parallel have local variables named `patterns`
# (the per-user pattern rows) that would shadow the module.
import patterns as patterns_mod

load_dotenv()

# ----------------------------------------------------------------------
# Groq client setup
# ----------------------------------------------------------------------
# Set GROQ_API_KEY in a .env file next to this script, or as an environment
# variable. Get a free key at https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
client = None

if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("[brain] Groq client initialised successfully")
    except Exception as e:
        print(f"[brain] Failed to initialise Groq client: {e}")
else:
    print("[brain] WARNING: No Groq API key configured.")

# ----------------------------------------------------------------------
# Static knowledge: personality + mood instructions, language names
# ----------------------------------------------------------------------

PERSONALITY_INSTRUCTIONS = {
    "friendly": "Be warm, conversational, supportive and approachable. Speak like a close friend who genuinely cares.",
    "professional": "Be formal, concise and precise. Avoid casual language and get straight to the point.",
    "motivational": "Be energetic, encouraging and positive, like a motivational coach pushing the user toward action.",
}

# Action-oriented mood handling: mood must change WHAT ARIA does, not just
# how she sounds. Every negative-mood directive demands ONE concrete offer,
# and all of them require using the injected context (profile, history,
# frequent topics) so support is specific, never generic sympathy.
MOOD_RESPONSE_INSTRUCTIONS = {
    "stressed": ("The user is stressed. Reference the ACTUAL task or topic they're stressed about "
                 "(from their message, recent history, or known patterns — never a generic 'that sounds tough'). "
                 "Offer ONE concrete small win: break the task into a first 10-minute step, offer to set a "
                 "reminder (TASK: line), or offer a relevant search/video."),
    "anxious": ("The user is anxious. Acknowledge the specific worry in their own words, then offer ONE "
                "concrete grounding action: a short break, writing the worry down as a task, or gently "
                "asking one focused question about it. Not a lecture."),
    "sad": ("The user is sad. Acknowledge genuinely and SPECIFICALLY what they said (no stock sympathy), "
            "then offer ONE concrete action: a video or search tied to one of their known interests "
            "(YOUTUBE:/SEARCH: line), a short break, or gently asking what's going on. Pick one, don't list options."),
    "tired": ("The user is tired. Keep it short and soft. Offer ONE concrete thing: suggest an actual break "
              "now, offer to defer their pending tasks to tomorrow, or a low-effort video tied to a known interest."),
    "frustrated": ("The user is frustrated. Name the specific thing frustrating them (from message/history), "
                   "acknowledge it plainly, then offer ONE practical unblock: break it down, try a different "
                   "approach, or park it with a reminder for later."),
    "happy": ("The user is happy. Engage with the SPECIFIC thing they're happy about and ask one real "
              "follow-up question about it — not generic upbeat filler."),
    "excited": ("The user is excited. Match it, engage with the specific thing, and ask a real follow-up "
                "that shows you understood exactly what they're excited about."),
    "confused": ("The user seems confused. Slow down. Check what part lost them, and offer to explain it "
                 "DIFFERENTLY (an analogy, an example) — never repeat the same explanation more softly."),
    "calm": "The user seems calm. Respond naturally and conversationally.",
    "calm confident": "The user seems calm and confident. Respond in a steady, capable tone.",
    "engaged": "The user seems engaged and attentive. Match their energy with thoughtful, substantive responses.",
    "distracted": "The user seems distracted. Be clear and concise to help them stay focused.",
    "surprised": "The user seems surprised. Acknowledge it naturally.",
}

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "yo": "Yoruba", "ig": "Igbo", "pcm": "Nigerian Pidgin", "ar": "Arabic",
    "hi": "Hindi", "pt": "Portuguese", "it": "Italian", "zh-cn": "Chinese",
    "ru": "Russian", "sw": "Swahili", "tr": "Turkish",
}

MOOD_DESCRIPTIONS = {
    "happy": "in good spirits", "sad": "a bit down", "stressed": "under some pressure",
    "anxious": "a little uneasy", "calm": "calm and steady", "tired": "running low on energy",
    "frustrated": "frustrated", "excited": "excited and energetic", "engaged": "engaged and attentive",
    "distracted": "a little distracted", "surprised": "surprised", "calm confident": "calm and confident",
}

NEGATIVE_TEXT_MOODS = {"sad", "angry", "frustrated", "stressed", "anxious", "disgusted"}


# ----------------------------------------------------------------------
# Mood fusion
# ----------------------------------------------------------------------

def fuse_moods(voice_mood, face_mood, text_mood, voice_pitch, fatigue, engagement):
    """
    Combine voice (25%), face (40%) and text (35%) mood signals into one
    fused mood via weighted voting, with physiological overrides for
    cases the vote alone would miss.

    Returns dict: {mood, intensity, description}
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

    description = f"User appears {MOOD_DESCRIPTIONS.get(fused_mood, fused_mood)}."
    return {"mood": fused_mood, "intensity": intensity, "description": description}


def analyze_text_mood(text):
    """
    Calls Groq with a small, fast prompt to classify the emotional tone of a
    message. Returns dict: {mood, intensity, emotions}. Falls back to a
    neutral 'calm' reading on any error (no client, parse failure, API error).
    """
    fallback = {"mood": "calm", "intensity": 0.5, "emotions": []}
    if client is None or not text:
        return fallback

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an emotion analysis engine. "
                                               "Respond with ONLY valid JSON, no markdown, no explanation."},
                {"role": "user", "content": (
                    'Analyze the emotional tone of this message and respond with exactly this JSON shape: '
                    '{"mood": "<one word>", "intensity": <0.0-1.0>, "emotions": ["..."]}\n\n'
                    f'Message: "{text}"'
                )},
            ],
            max_tokens=80,
            temperature=0.3,
        )
        raw = completion.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return fallback

        import json
        data = json.loads(match.group(0))
        mood = str(data.get("mood", "calm")).strip().lower()
        intensity = float(data.get("intensity", 0.5))
        emotions = data.get("emotions", [])
        if not isinstance(emotions, list):
            emotions = []
        return {"mood": mood or "calm", "intensity": min(max(intensity, 0.0), 1.0), "emotions": emotions}
    except Exception as e:
        print(f"[brain] analyze_text_mood failed: {e}")
        return fallback


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

def build_prompt(user_message, user_id, fused_mood, intensity, language, personality_mode,
                  context, patterns, tasks, face_emotion, fatigue, engagement,
                  voice_pitch, voice_speed):
    """Build the full, context-rich prompt sent to the LLM for the main response."""
    personality_instruction = PERSONALITY_INSTRUCTIONS.get(personality_mode, PERSONALITY_INSTRUCTIONS["friendly"])
    mood_instruction = MOOD_RESPONSE_INSTRUCTIONS.get(fused_mood, "Respond naturally and warmly.")
    language_name = LANGUAGE_NAMES.get(language, "English")

    sections = [
        "You are ARIA, a personal AI assistant having a natural voice conversation with your user.",
        f"PERSONALITY MODE: {personality_mode.upper()}\n{personality_instruction}",
        (
            "Physical state detected:\n"
            f"Face emotion: {face_emotion}\n"
            f"Fatigue level: {fatigue}/1.0\n"
            f"Engagement: {engagement}/1.0\n"
            f"Voice pitch: {voice_pitch}Hz\n"
            f"Voice speed: {voice_speed} words/sec\n"
            f"Fused mood: {fused_mood} (intensity: {intensity})"
        ),
        f"MOOD RESPONSE GUIDANCE: {mood_instruction}",
        ("MOOD-CONTEXT CONNECTION: when following the mood guidance above, draw on the "
         "KNOWN PATTERNS, KNOWN INTERESTS, learned profile and RECENT CONVERSATION HISTORY "
         "sections of this prompt so acknowledgements and offers are specific "
         "(e.g. 'I know the project deadline has been weighing on you') — never generic."),
    ]

    # Ranked interests for personalised suggestions (Part of the behavioural-
    # learning story: cheer-up offers tie to what the user actually cares about).
    try:
        topics = database.get_patterns_by_type(user_id, "frequent_topic", limit=5)
        apps   = database.get_patterns_by_type(user_id, "app_opened", limit=3)
        interests = [f"{p['pattern_value']} (mentioned {p['frequency']}x)" for p in topics]
        interests += [f"uses the app '{p['pattern_value']}' often ({p['frequency']}x)" for p in apps]
        if interests:
            sections.append("KNOWN INTERESTS / FREQUENT TOPICS, ranked by frequency: "
                            + "; ".join(interests))
        else:
            sections.append("KNOWN INTERESTS: none learned yet (new user) — when a suggestion "
                            "would help, fall back to a sensible generic mood-appropriate one "
                            "rather than inventing an interest.")
    except Exception as e:
        print(f"[brain] interests lookup failed: {e}")

    if context:
        history_lines = []
        for turn in context:
            history_lines.append(f"User: {turn.get('user_message', '')}")
            history_lines.append(f"ARIA: {turn.get('ai_response', '')}")
        sections.append("RECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines))

    if patterns:
        pattern_lines = [
            f"- {p.get('pattern_type')}: {p.get('pattern_value')} (seen {p.get('frequency')} times)"
            for p in patterns
        ]
        sections.append("KNOWN PATTERNS ABOUT THIS USER:\n" + "\n".join(pattern_lines))

    # Long-term behavioural profile (cached, refreshed after every turn) —
    # this is what separates a user with weeks of history from a new user.
    try:
        profile = database.get_behavioural_profile(user_id)
        if profile and profile.get("summary"):
            sections.append(
                "WHAT YOU HAVE LEARNED ABOUT THIS USER OVER TIME "
                "(weave in naturally when relevant, never recite):\n" + profile["summary"]
            )
    except Exception as e:
        print(f"[brain] behavioural profile lookup failed: {e}")

    # Learned response-style preference (from the user's own past feedback).
    try:
        style_prefs = database.get_patterns_by_type(user_id, "style_pref", limit=1)
        if style_prefs and style_prefs[0].get("frequency", 0) >= 2:
            pref = style_prefs[0]["pattern_value"]
            if pref == "detailed":
                sections.append("LEARNED STYLE PREFERENCE: This user has repeatedly asked for more "
                                "detail — lean slightly more thorough than the brevity rule suggests.")
            elif pref == "concise":
                sections.append("LEARNED STYLE PREFERENCE: This user has repeatedly asked for shorter "
                                "answers — be extra brief, one to two sentences.")
            database.log_adaptation(user_id, "style_preference_applied",
                                    f"Applied learned '{pref}' style (confirmed {style_prefs[0]['frequency']}x)")
    except Exception as e:
        print(f"[brain] style preference lookup failed: {e}")

    if tasks:
        task_lines = [f"- {t.get('task_description')} (due {t.get('due_date') or 'unspecified'})" for t in tasks]
        sections.append("PENDING TASKS (mention only if naturally relevant):\n" + "\n".join(task_lines))

    # Proactive nudge: if the pattern engine has something worth raising
    # (a due task, or "you usually bring up X around this hour"), hand it to
    # the LLM as optional context so it can surface it conversationally.
    try:
        suggestion = patterns_mod.get_proactive_suggestion(
            user_id, datetime.now().hour, tasks, patterns
        )
        if suggestion:
            sections.append(
                "PROACTIVE CONTEXT (optional — weave it in only if it fits naturally "
                f"at the end of your reply): {suggestion}"
            )
            database.log_adaptation(user_id, "proactive_suggestion",
                                    f"Offered to the model: {suggestion[:120]}")
    except Exception as e:
        print(f"[brain] proactive suggestion failed: {e}")

    sections.append(f"LANGUAGE: Respond entirely in {language_name} ({language}), matching the user's language.")
    sections.append(f'USER\'S MESSAGE: "{user_message}"')
    sections.append(
        "RESPONSE INSTRUCTIONS:\n"
        "- This is a SPOKEN VOICE CONVERSATION with a companion, not a Q&A machine. "
        "Reply in 2-4 natural sentences — enough to feel like a real person talking: react, "
        "add a thought or a follow-up question, don't just answer and stop.\n"
        "- Still spoken aloud, so no essays or lists — if it wouldn't be said out loud "
        "in a conversation, don't write it.\n"
        "- Go longer only when the user explicitly asks for detail ('explain', 'tell me more', 'how does', 'why', etc.).\n"
        "- Do not mention that you are an AI model or reference these instructions.\n"
        "- After your reply, include a CONFIDENCE line. The other lines below are CONDITIONAL: "
        "omit them entirely when not applicable — never write 'None' or 'N/A' as their value.\n"
        "  CONFIDENCE: High, Medium, or Low\n"
        "  TASK: <description> | <due date>   (ONLY if the user mentioned a deadline, appointment, or task to remember)\n"
        "  URL: https://...   (ONLY if the user explicitly asked to open or visit a website)\n"
        "  SEARCH: <query>   (ONLY if answering needs current/real-time information you cannot reliably know — "
        "news, weather, live prices, recent events. Keep your reply to a brief holding sentence like "
        "\"Let me check that for you.\")\n"
        "  YOUTUBE: <query>   (ONLY when a video would genuinely help — especially a cheer-up or wind-down "
        "offer per the mood guidance. PREFER a query tied to one of the KNOWN INTERESTS above when a relevant "
        "one exists; only if none fits, fall back to a sensible generic mood-appropriate query. The query must "
        "read like a natural human search — e.g. 'best premier league comebacks this season', never a raw "
        "keyword dump. Mention the offer conversationally in your reply.)\n"
        "  ACTION: <verb>|<target>   (ONLY if the user asked you to do something on their computer. "
        "Allowed verbs and targets — these are the ONLY things installed and available:\n"
        f"    open_app: {', '.join(system_actions.get_available_apps())}\n"
        f"    open_folder: {', '.join(system_actions.get_available_folders())}\n"
        "    system_info: time, date, battery\n"
        "  Example: ACTION: open_app|chrome\n"
        "  If the user asks to open an app or folder NOT in these lists, do NOT emit any ACTION line "
        "at all — NEVER open a substitute app the user didn't ask for. Refuse in text only: say you can "
        "only open a specific set of apps right now and name 3-4 from the open_app list as examples.)"
    )

    return "\n\n".join(sections)


# ----------------------------------------------------------------------
# Response generation
# ----------------------------------------------------------------------

def get_ai_response(prompt):
    """Call Groq for the main conversational response. Retries on rate limits, never raises."""
    if client is None:
        return "I'm currently unable to reach my reasoning engine - my Groq API key isn't configured.\nCONFIDENCE: Low"

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                # 320: room for a 2-4 sentence conversational reply plus the
                # CONFIDENCE/TASK/URL metadata lines without truncating them.
                # Length is shaped by the prompt rule, not the hard cap.
                max_tokens=320,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except RateLimitError:
            wait = 2 ** attempt
            print(f"[brain] Rate limited by Groq, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"[brain] get_ai_response failed: {e}")
            break

    return "I'm having trouble forming a response right now, but I'm still here with you.\nCONFIDENCE: Low"


# ----------------------------------------------------------------------
# Web search (DuckDuckGo, no API key)
# ----------------------------------------------------------------------

def web_search(query, max_results=4):
    """
    Search DuckDuckGo and return a compact text digest of the top results,
    or None on any failure (offline, blocked network, package missing).
    """
    try:
        try:
            from ddgs import DDGS  # current package name
        except ImportError:
            from duckduckgo_search import DDGS  # pre-rename fallback
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        lines = []
        for r in results:
            title = (r.get("title") or "").strip()
            body  = (r.get("body") or "").strip()[:250]
            if title or body:
                lines.append(f"- {title}: {body}")
        return "\n".join(lines) if lines else None
    except Exception as e:
        print(f"[brain] web_search failed: {e}")
        return None


def resolve_search_if_needed(ai_response, user_message, language="en"):
    """
    If the model requested a web search (SEARCH: line), run it and generate a
    final grounded reply. Returns (final_response, searched: bool).
    On search failure, strips the SEARCH line and returns an honest fallback.
    """
    query = extract_search(ai_response)
    if not query:
        return ai_response, False

    print(f"[brain] Model requested search: {query!r}")
    digest = web_search(query)
    if digest is None:
        fallback = ("I tried to look that up but couldn't reach the web just now — "
                    "I can answer from what I already know if you'd like.\nCONFIDENCE: Low")
        return fallback, True

    language_name = LANGUAGE_NAMES.get(language, "English")
    followup = (
        "You are ARIA, a personal AI assistant in a spoken voice conversation.\n\n"
        f'The user asked: "{user_message}"\n\n'
        f"You searched the web and found:\n{digest}\n\n"
        "RESPONSE INSTRUCTIONS:\n"
        "- Answer the user's question using these results, in 2-3 spoken sentences maximum.\n"
        "- Do not read out URLs or mention that you performed a search unless it flows naturally.\n"
        f"- Respond in {language_name}.\n"
        "- After your reply, on a new line include: CONFIDENCE: High, Medium, or Low"
    )
    final = get_ai_response(followup)
    return final, True


def _time_greeting(time_of_day):
    return {
        "morning": "Good morning",
        "afternoon": "Good afternoon",
        "evening": "Good evening",
        "night": "Good evening",
    }.get(time_of_day, "Hello")


def get_greeting(user_id, name, face_emotion=None, fatigue=0, time_of_day=""):
    """
    Generate a short, personalised greeting by pulling the user's real
    history (last conversation, pending tasks, top pattern, total session
    count) from the database and handing it to Groq. Falls back to a simple
    static greeting if the LLM is unavailable or the call fails.
    """
    total = database.get_total_conversations(user_id)
    print(f"[greeting] Total conversations for user {user_id}: {total}")
    history = database.get_conversation_history(user_id, limit=1)
    tasks = database.get_pending_tasks(user_id)
    # Belt-and-braces against the old "TASK: None" placeholder bug: never let
    # junk rows inflate the count the greeting reports.
    tasks = [t for t in tasks
             if (t.get("task_description") or "").strip()
             and not _is_placeholder(t["task_description"])]
    pats = database.get_patterns(user_id, limit=3)

    # Stated as an explicit FACT either way: omitting the tasks line entirely
    # lets the model invent a number ("you have 9 pending tasks") instead of
    # knowing there are none.
    if tasks:
        task_text = f"They have {len(tasks)} pending task(s) including: {tasks[0]['task_description'][:50]}"
    else:
        task_text = "FACT: They have NO pending tasks. Do not mention tasks at all."

    pattern_text = ""
    if pats:
        top = pats[0]
        pattern_text = f"Their most common pattern is {top['pattern_type']}: {top['pattern_value']}"

    history_text = ""
    if history:
        last = history[0]
        history_text = f"Last time they said: '{last['user_message'][:80]}'"

    face_text = ""
    if face_emotion and face_emotion != "neutral":
        face_text = f"Their face currently shows: {face_emotion}"

    fatigue_text = ""
    if fatigue and fatigue > 0.6:
        fatigue_text = "They appear tired."

    if not time_of_day:
        hour = datetime.now().hour
        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

    first_time = (total is None or total == 0)

    if first_time:
        prompt = f"""Generate a warm friendly first-time greeting for {name} who is using ARIA for the very first time.
It is {time_of_day}.
Respond in ONE short, natural sentence. Maximum 15-20 words.
This is spoken aloud on app startup — brevity is critical.
Do not mention you are an AI."""
    else:
        prompt = f"""Generate a personalised returning-user greeting for {name}.
It is {time_of_day}.
They have had {total} conversations with ARIA.
{history_text}
{task_text}
{pattern_text}
{face_text}
{fatigue_text}
Rules:
- Respond in ONE short, natural sentence. Maximum 15-20 words.
- This is spoken aloud on app startup — brevity is critical. Something you'd
  naturally say walking into a room, not a paragraph.
- Weave in AT MOST one personal detail from above (a real pending task takes priority).
- Match the time of day naturally
- Never say the same thing twice
- Do not mention you are an AI"""

    fallback = (
        f"Hello {name}! Welcome to ARIA. I am your personal AI assistant and I am here to learn and adapt to you over time."
        if first_time
        else f"Welcome back {name}! Good {time_of_day}. Great to see you again."
    )

    if client is None:
        return fallback

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            # Hard backstop for the one-sentence rule: ~45 tokens ≈ 30 words.
            # (The main conversation call keeps its own separate 220 cap.)
            max_tokens=45,
            temperature=0.9,
        )
        greeting = response.choices[0].message.content.strip()
        if not first_time:
            used = [s for s, present in [("last conversation", bool(history_text)),
                                          ("pending tasks", bool(task_text)),
                                          ("top pattern", bool(pattern_text))] if present]
            if used:
                database.log_adaptation(user_id, "greeting_personalised",
                                        "Greeting drew on: " + ", ".join(used))
        return greeting
    except Exception as e:
        print(f"[brain] get_greeting failed: {e}")
        return fallback


def run_parallel(user_message, user_id, voice_features, face_features, language):
    """
    Runs text mood analysis and the main AI response generation concurrently
    to keep latency low. The response prompt is built from voice+face mood
    signals (already available instantly) rather than waiting on the LLM-based
    text mood analysis, so the two calls can genuinely run side by side -
    they finish at roughly the same time instead of one blocking the other.

    Returns dict: {ai_response, text_mood, prompt_mood}
    """
    voice_features = voice_features or {}
    face_features = face_features or {}

    user = database.get_user(user_id)
    personality_mode = user.get("personality_mode", "friendly") if user else "friendly"
    context = database.get_conversation_history(user_id, limit=3)
    patterns = database.get_patterns(user_id, limit=5)
    tasks = database.get_pending_tasks(user_id)

    voice_mood = voice_features.get("mood", "calm")
    voice_pitch = voice_features.get("pitch", 0.0)
    voice_speed = voice_features.get("speaking_speed", 0.0)
    face_emotion = face_features.get("emotion", "neutral")
    fatigue = face_features.get("fatigue", 0.0)
    engagement = face_features.get("engagement", 0.5)
    face_mood = face_features.get("fused_mood", "calm")

    # Personal KNN classifier: once trained on this user's own history, its
    # prediction replaces the generic rule-based voice mood — the moment the
    # system stops guessing from fixed thresholds and starts using what it
    # learned about *this* user's voice/face signature.
    knn_mood = patterns_mod.predict_mood_knn({
        "pitch": voice_pitch, "speaking_speed": voice_speed,
        "fatigue": fatigue, "engagement": engagement,
    })
    if knn_mood and voice_pitch > 0:
        voice_mood = knn_mood
        try:
            database.log_adaptation(user_id, "knn_mood_used",
                                    f"Personal classifier read this turn as '{knn_mood}' "
                                    f"(pitch {voice_pitch:.0f}Hz, {voice_speed:.1f} w/s)")
        except Exception:
            pass

    preliminary_fusion = fuse_moods(voice_mood, face_mood, "calm", voice_pitch, fatigue, engagement)

    prompt = build_prompt(
        user_message, user_id, preliminary_fusion["mood"], preliminary_fusion["intensity"],
        language, personality_mode, context, patterns, tasks,
        face_emotion, fatigue, engagement, voice_pitch, voice_speed,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        text_mood_future = executor.submit(analyze_text_mood, user_message)
        response_future = executor.submit(get_ai_response, prompt)
        text_mood_result = text_mood_future.result()
        ai_response = response_future.result()

    return {
        "ai_response": ai_response,
        "text_mood": text_mood_result,
        "prompt_mood": preliminary_fusion,
    }


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------

# Models sometimes fill conditional metadata lines with a placeholder instead
# of omitting them ("SEARCH: None") — treat those as absent.
_PLACEHOLDER_VALUES = {"none", "n/a", "null", "-", "nil", "'none'", '"none"'}


def _is_placeholder(value: str) -> bool:
    return value.strip().strip("'\"").lower() in _PLACEHOLDER_VALUES


def extract_task(response):
    """Parse a 'TASK: description | due_date' line. Returns dict or None."""
    match = re.search(r"^TASK:\s*(.+)$", response, re.MULTILINE | re.IGNORECASE)
    if not match or _is_placeholder(match.group(1)):
        return None
    content = match.group(1).strip()
    if "|" in content:
        desc, _, due = content.partition("|")
        due = due.strip()
        return {"description": desc.strip(), "due_date": due or None}
    return {"description": content, "due_date": None}


def extract_url(response):
    """Parse a 'URL: https://...' line. Returns the URL string or None."""
    match = re.search(r"^URL:\s*(\S+)$", response, re.MULTILINE | re.IGNORECASE)
    if not match or _is_placeholder(match.group(1)):
        return None
    url = match.group(1).strip()
    return url if url.lower().startswith(("http://", "https://")) else None


def extract_confidence(response):
    """Parse a 'CONFIDENCE: High/Medium/Low' line. Defaults to 'Medium'."""
    match = re.search(r"^CONFIDENCE:\s*(High|Medium|Low)", response, re.MULTILINE | re.IGNORECASE)
    return match.group(1).capitalize() if match else "Medium"


def extract_search(response):
    """Parse a 'SEARCH: query' line. Returns the query string or None."""
    match = re.search(r"^SEARCH:\s*(.+)$", response, re.MULTILINE | re.IGNORECASE)
    if not match or _is_placeholder(match.group(1)):
        return None
    return match.group(1).strip()


def extract_youtube(response):
    """Parse a 'YOUTUBE: query' line. Returns the search query string or None."""
    match = re.search(r"^YOUTUBE:\s*(.+)$", response, re.MULTILINE | re.IGNORECASE)
    if not match or _is_placeholder(match.group(1)):
        return None
    return match.group(1).strip()


def extract_action(response):
    """Parse an 'ACTION: verb|target' line. Returns dict {verb, target} or None."""
    match = re.search(r"^ACTION:\s*(\w+)\s*\|\s*(\S+)", response, re.MULTILINE | re.IGNORECASE)
    if not match or _is_placeholder(match.group(1)):
        return None
    return {"verb": match.group(1).strip().lower(), "target": match.group(2).strip().lower()}


def clean_response(response):
    """Strip CONFIDENCE:/TASK:/URL:/SEARCH:/ACTION:/YOUTUBE: lines, returning clean text suitable for speaking."""
    cleaned = re.sub(r"^(CONFIDENCE|TASK|URL|SEARCH|ACTION|YOUTUBE):.*$", "", response, flags=re.MULTILINE | re.IGNORECASE)
    return "\n".join(line for line in cleaned.splitlines() if line.strip()).strip()


if __name__ == "__main__":
    # Quick manual smoke test: python brain.py
    print("Testing mood fusion...")
    print(fuse_moods("calm", "happy", "calm", 150.0, 0.1, 0.8))
    print(fuse_moods("stressed", "angry", "frustrated", 220.0, 0.2, 0.5))
    print(fuse_moods("calm", "calm", "calm", 150.0, 0.85, 0.5))

    print("Testing response parsing...")
    sample = "I hear you! Let's tackle that deadline together.\nCONFIDENCE: High\nTASK: Submit report | 2026-07-01\nURL: https://example.com"
    print("confidence:", extract_confidence(sample))
    print("task:", extract_task(sample))
    print("url:", extract_url(sample))
    print("clean:", repr(clean_response(sample)))

    print("Client configured:", client is not None)
    if client is not None:
        print("Testing live Groq call...")
        print(get_ai_response("Say hello in one short sentence.\nCONFIDENCE: High"))
