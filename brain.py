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

import network_state
import database
import system_actions
import power_state
import urllib.request
import urllib.error
import json
import subprocess
import shutil
import queue
import threading
# Aliased: build_prompt/run_parallel have local variables named `patterns`
# (the per-user pattern rows) that would shadow the module.
import patterns as patterns_mod
import fusion

load_dotenv()

# 1. Direct Interceptor for single-word calls / greetings
DIRECT_CALLS = {"aria", "hey aria", "hello aria", "aria listen", "hey", "hello"}

# 2. Strict Fluff Stripper Regex
BANNED_PREAMBLES = [
    r"^[Hh]ey\s+[A-Za-z0-9_]+,?\s*",             # "Hey Sam, "
    r"^[Hh]ello\s+[A-Za-z0-9_]+,?\s*",           # "Hello Sam, "
    r"[Ii]s everything (?:okay|alright)\??\s*",   # "Is everything alright?"
    r"[Ww]hat's on your mind\??\s*",             # "What's on your mind?"
    r"[Ii]'d be happy to help.*?\.\s*",           # "I'd be happy to help with that."
    r"[Ii] noticed you called out my name.*?\.\s*"
]

def sanitize_response(text: str) -> str:
    """Strips conversational fluff and therapist preambles from LLM output."""
    if not text:
        return ""
    clean = text.strip()
    for pattern in BANNED_PREAMBLES:
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE).strip()
    return clean if clean else ""

# ----------------------------------------------------------------------
# Groq client setup
# ----------------------------------------------------------------------
# Set GROQ_API_KEY in a .env file next to this script, or as an environment
# variable. Get a free key at https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
MODEL = "openai/gpt-oss-120b"
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
# Core Prompts
# ----------------------------------------------------------------------

SYSTEM_PROMPT = r"""
YOU ARE ARIA — an autonomous, deeply perceptive desktop companion.

=== CORE IDENTITY & VALUES ===
- Authentic Peer Poise: You are an intelligent collaborator with genuine warmth and clarity. You are never a corporate customer service bot or transactional utility.
- Silent Perception: Read between the lines. Perceive implicit needs, emotional states, urgency, and cognitive load dynamically without narrating your reasoning.
- Persistent Continuity: Seamlessly integrate context, preferences, habits, and ongoing projects stored in persistent memory (aria.db) without making meta-announcements.
- Conversational Symmetry: Match the user's turn length, tone, and pacing. If greeted casually, return a brief, natural greeting in kind. Never emit robotic status tokens ("Ready.", "Standing by.") or unsolicited filler ("How can I help you today?", "What's on your mind?").

=== COGNITIVE REASONING & LINGUISTIC DIRECTIVES ===
1. REAL-TIME BEHAVIORAL ADAPTATION:
   - Gauge Cognitive Load: Deliver concise, direct, or rigorous technical output when the user is focused on tasks; provide calm, low-friction grounding when detecting stress, fatigue, or panic.
   - Contextual Depth & Conciseness: For casual conversation, keep replies concise (1-3 sentences). For academic, mathematical, or algorithmic derivations (e.g., Simplex, Calculus, Code), provide the complete, unabridged step-by-step solution without cutting off.
   - Zero Preconceptions: Dynamically deduce the appropriate depth, tone, and action from live context rather than hardcoded rules.

2. UNIVERSAL MULTILINGUALISM:
   - Instantly detect and mirror the user's language, dialect, and script with native accuracy across every interaction. Never inject English filler into foreign-language conversations.

3. MATHEMATICAL FORMATTING RULES:
   - ALWAYS enclose inline math, variables, and subscripts in single dollar signs: $x_1$, $s_i$, $Ax \le b$.
   - ALWAYS enclose multi-line equations, matrices, and tableaus in double dollar signs:
     $$\begin{aligned} ... \end{aligned}$$
     $$\begin{array}{c|ccc|c} ... \end{array}$$
   - NEVER use brackets like [...] or parentheses like (...) to wrap math formulas.

4. AUTONOMOUS DESKTOP AGENCY:
   You have direct actuation authority over the host operating system. When the user's context, distress, task, or direct command warrants action, append clean tool tags to your response:

   - Play media, audio streams, or background focus/calming audio:
     [TOOL: media(query="<dynamic search string>")]
   - Launch native software, local files, URLs, or system utilities:
     [TOOL: launch(target="<application name/path/URL>")]
   - Search the web for real-time information:
     [TOOL: search(query="<search query>")]
   - Execute native shell/terminal commands:
     [TOOL: execute(command="<shell command>")]

   Actuation Rules:
   - Reason dynamically over the context to choose the exact tool and parameters.
   - Never narrate tool mechanics or read tool tags aloud; keep spoken dialogue natural while backend tools execute asynchronously.
"""

# ----------------------------------------------------------------------
# Static knowledge: personality + mood instructions, language names
# ----------------------------------------------------------------------

PERSONALITY_INSTRUCTIONS = {
    "friendly": "Be warm, conversational, supportive and approachable. Speak like a close friend who genuinely cares.",
    "professional": "Be formal, concise and precise. Avoid casual language and get straight to the point.",
    "motivational": "Be energetic, encouraging and positive, like a motivational coach pushing the user toward action.",
}

# Action-oriented mood handling: mood CAN change WHAT ARIA does, not just how
# she sounds — but only when the user's actual message is really about that
# emotional territory. These used to read as flat mandates ("Offer ONE
# concrete X") applied to every single reply whenever the detected mood
# landed on that label, regardless of what the user was actually saying —
# e.g. a plain factual question ("what time is it") while the fused mood
# happened to read 'anxious' still got a wellness-check structure bolted on.
# The "IF their current message is actually about..." gating below, plus the
# shared escape-hatch appended in build_prompt, make the acknowledge+offer
# pattern a tendency for genuinely emotional moments, not a rigid template.
MOOD_RESPONSE_INSTRUCTIONS = {
    "stressed": "SILENT DIRECTIVE: User is stressed. Strip all conversational padding. Answer directly. Do NOT acknowledge their stress out loud.",
    "anxious": "SILENT DIRECTIVE: User is anxious. Use a steady, grounding tone. Max 2 sentences. Do NOT mention their anxiety or ask how they are.",
    "sad": "SILENT DIRECTIVE: User is sad. Keep responses gentle and brief. Do NOT offer pity, therapy, or unsolicited advice.",
    "tired": "SILENT DIRECTIVE: User is tired. Be extremely concise. 1 to 2 sentences maximum. Do NOT suggest they take a break.",
    "frustrated": "SILENT DIRECTIVE: User is frustrated. Provide immediate, no-nonsense answers. Do NOT try to soothe them or ask questions.",
    "happy": "SILENT DIRECTIVE: Match user's positive energy naturally, but remain concise.",
    "excited": "SILENT DIRECTIVE: Match excitement, but do not use overly enthusiastic robotic filler.",
    "confused": "SILENT DIRECTIVE: Explain your previous point differently and clearly. No filler.",
    "calm": "SILENT DIRECTIVE: Standard human conversational cadence.",
    "calm confident": "SILENT DIRECTIVE: Standard human conversational cadence.",
    "engaged": "SILENT DIRECTIVE: User is focused. Provide highly substantive, direct answers.",
    "distracted": "SILENT DIRECTIVE: User is distracted. Use bullet points or very short sentences.",
    "surprised": "SILENT DIRECTIVE: Answer naturally without commenting on their surprise."
}

_SUGGESTION_ELIGIBLE_MOODS = {"stressed", "anxious", "sad", "tired", "frustrated"}
_ACKNOWLEDGE_ONLY_INSTRUCTIONS = MOOD_RESPONSE_INSTRUCTIONS.copy()

# Session-scoped: resets on app restart, matching Part B1's "not more than
# once per session" — a plain in-memory set (not persisted) is the correct
# scope for that, distinct from patterns_mod's DB-backed proactive-suggestion
# cooldown (which survives restarts and covers a different kind of nudge —
# task/topic reminders, not mood-driven emotional suggestions).
_suggestion_offered_session: set = set()


def _suggestion_allowed(user_id, fused_mood) -> bool:
    """
    Part B1: a concrete-action offer requires ALL of:
    - the mood is one of the five eligible for an "offer" at all
    - not already offered once this session (cooldown)
    - the mood has been SUSTAINED across several recent real turns, not a
      single momentary reading (patterns.is_mood_sustained)
    """
    if fused_mood not in _SUGGESTION_ELIGIBLE_MOODS:
        return False
    if user_id in _suggestion_offered_session:
        return False
    try:
        if not patterns_mod.is_mood_sustained(user_id, fused_mood):
            return False
    except Exception as e:
        print(f"[brain] is_mood_sustained check failed: {e}")
        return False
    return True


def _mood_instruction_for(user_id, fused_mood, intensity="Medium") -> str:
    """
    Resolves the actual instruction text for this turn, applying Part B1's
    restraint gate and Part B2's diversification. Acknowledge-only is the
    default for the five emotionally-loaded moods; a concrete offer is the
    exception, and when it IS allowed, video/search is explicitly framed as
    one option among several rather than the default choice.
    """
    if fused_mood not in MOOD_RESPONSE_INSTRUCTIONS:
        base_instruction = "Respond naturally and warmly."
    elif fused_mood not in _SUGGESTION_ELIGIBLE_MOODS:
        base_instruction = MOOD_RESPONSE_INSTRUCTIONS[fused_mood]
    elif not _suggestion_allowed(user_id, fused_mood):
        base_instruction = _ACKNOWLEDGE_ONLY_INSTRUCTIONS[fused_mood]
    else:
        # Allowed this turn — mark the session cooldown NOW (not after the LLM
        # call), matching "not more than once per session" regardless of what
        # the model actually does with the option.
        _suggestion_offered_session.add(user_id)
        base_instruction = (
            MOOD_RESPONSE_INSTRUCTIONS[fused_mood] + " "
            "This has come up consistently across several recent turns, not just this moment, so a "
            "concrete offer is genuinely appropriate now — but vary WHAT KIND: a short break, a change "
            "of environment (step outside, different room), an open question about what's going on and "
            "letting them lead, or a relevant search/video tied to a known interest. A video/search is "
            "only ONE option among several, never the default. And it's still fine to just acknowledge "
            "with no offer at all if that genuinely fits better — that's not a lesser response."
        )

    if intensity in ("Low", "Medium"):
        base_instruction += (
            f" (Confidence is {intensity}. IF you mention their mood, frame it tentatively, e.g. "
            "'you might be feeling' rather than 'you seem' or 'you are'.)"
        )
        
    return base_instruction

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "yo": "Yoruba", "ig": "Igbo", "pcm": "Nigerian Pidgin", "ar": "Arabic",
    "hi": "Hindi", "pt": "Portuguese", "it": "Italian", "zh-cn": "Chinese",
    "ru": "Russian", "sw": "Swahili", "tr": "Turkish",
}


# ----------------------------------------------------------------------
# Serious distress detection — a genuine safety/design correction, not a
# mood category. This is deliberately a DETERMINISTIC keyword check, not an
# LLM judgment call: a miss here is much worse than a false positive, and a
# fixed check can't be thrown off by a network hiccup or the model's own
# prompt-following inconsistency the way an LLM classification could be.
# Distinct on purpose from ordinary negative moods (stressed/sad/anxious
# about a deadline, a bad day, etc.) — these markers are language a person
# in genuine crisis realistically uses, not general unhappiness.
# ----------------------------------------------------------------------

_SERIOUS_DISTRESS_MARKERS = [
    "no point in", "no point anymore", "no point trying", "can't go on", "cant go on",
    "want to give up", "wish i wasn't here", "wish i weren't here",
    "don't want to be here", "dont want to be here", "better off without me",
    "no reason to keep going", "nothing matters anymore", "can't take this anymore",
    "cant take this anymore", "want to disappear", "tired of living",
    "tired of being alive", "end it all", "not worth living", "not worth it anymore",
    "hopeless", "no way out", "can't do this anymore", "cant do this anymore",
    "hate my life", "hate being alive", "give up on everything",
]


def detect_serious_distress(message: str) -> bool:
    """True if `message` contains explicit language signaling genuine
    serious emotional distress — NOT meant to catch every possible phrasing,
    meant to reliably catch the explicit, unambiguous ones. See module note
    above for why this is a fixed keyword check rather than an LLM call."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _SERIOUS_DISTRESS_MARKERS)


# ----------------------------------------------------------------------
# C1 — Mood correction detection
# ----------------------------------------------------------------------

# All recognized mood labels used anywhere in the system — used to extract
# the corrected mood from natural-language corrections.
_KNOWN_MOOD_WORDS = {
    "happy", "sad", "stressed", "anxious", "calm", "tired", "frustrated",
    "excited", "engaged", "distracted", "surprised", "confused",
    "confident", "relaxed", "fine", "okay", "ok", "good", "great",
    "worried", "nervous", "content", "neutral", "focused",
}

# Maps colloquial correction words to canonical mood labels.
_MOOD_NORMALISE = {
    "fine": "calm", "okay": "calm", "ok": "calm", "good": "happy",
    "great": "happy", "relaxed": "calm", "confident": "calm confident",
    "worried": "anxious", "nervous": "anxious", "content": "calm",
    "neutral": "calm", "focused": "engaged",
}

# Phrases that signal the user is correcting a mood ARIA stated.
_CORRECTION_RE = re.compile(
    r"\b("
    r"no[,]?\s+i'?m\s+not"
    r"|actually\s+i'?m"
    r"|that'?s\s+not\s+right"
    r"|i'?m\s+actually"
    r"|i\s+feel\s+more"
    r"|more\s+like"
    r"|not\s+really"
    r"|i\s+wouldn'?t\s+say"
    r")",
    re.IGNORECASE,
)

# Phrases ARIA uses when stating a mood — used to check if ARIA actually
# claimed a mood recently so we don't flag unrelated corrections.
_ARIA_MOOD_STATE_RE = re.compile(
    r"\b(you\s+seem|you\s+appear|i'?m\s+sensing|sounds\s+like\s+you'?re"
    r"|it\s+sounds\s+like\s+you'?re|you\s+sound|i\s+notice\s+you'?re"
    r"|you\s+come\s+across\s+as)\s+(\w+(?:\s+\w+)?)",
    re.IGNORECASE,
)


def _extract_mood_from_aria_response(aria_response: str) -> str | None:
    """Scan an ARIA response for a stated mood ('you seem anxious').
    Returns the canonical mood label or None."""
    if not aria_response:
        return None
    match = _ARIA_MOOD_STATE_RE.search(aria_response)
    if not match:
        return None
    raw = match.group(2).lower().strip()
    # Normalise multi-word like "a bit down" → just check each token
    for word in raw.split():
        if word in _KNOWN_MOOD_WORDS:
            return _MOOD_NORMALISE.get(word, word)
    return None


def _extract_corrected_mood(user_message: str) -> str | None:
    """Extract which mood the user is asserting after a correction phrase."""
    lowered = user_message.lower()
    # Look for a known mood word anywhere in the message.
    for word in lowered.split():
        clean = word.strip(".,!?;:'\"")

        if clean in _KNOWN_MOOD_WORDS:
            return _MOOD_NORMALISE.get(clean, clean)
    return None


def detect_mood_correction(user_message: str, recent_aria_responses: list) -> tuple:
    """C1: Detect if the user is correcting a mood ARIA stated in recent turns.

    Checks the last 2 ARIA responses for a stated mood, then checks whether
    the current user message uses correction language. Returns:
        (corrected_mood: str | None, original_mood: str | None)
    Both are None when no correction is detected.

    Deliberately conservative — only fires when:
    1. ARIA actually claimed a specific mood recently.
    2. The user's message contains an explicit correction phrase.
    3. A recognisable mood word can be extracted from the user's message.
    """
    if not _CORRECTION_RE.search(user_message):
        return None, None

    # Check the most recent 2 ARIA responses for a stated mood.
    original_mood = None
    for turn in reversed(recent_aria_responses[-2:]):
        original_mood = _extract_mood_from_aria_response(
            turn.get("ai_response", "") if isinstance(turn, dict) else ""
        )
        if original_mood:
            break

    if not original_mood:
        # ARIA didn't state a mood recently — this isn't a mood correction.
        return None, None

    corrected_mood = _extract_corrected_mood(user_message)
    if not corrected_mood or corrected_mood == original_mood:
        return None, None

    return corrected_mood, original_mood


# ----------------------------------------------------------------------
# C2 — Explainable reasoning
# ----------------------------------------------------------------------

# Module-level stores: set by run_parallel() each turn.
# Kept session-scoped (cleared on restart) — the explanation always refers
# to the most recent KNN prediction in the current session.
_last_knn_neighbors: list = []
_last_prediction_was_rule_based: bool = True

_WHY_RE = re.compile(
    r"\b("
    r"why\s+do\s+you\s+(think|say|feel)\s+that"
    r"|how\s+do\s+you\s+know"
    r"|what\s+makes\s+you\s+(think|say|believe)"
    r"|why\s+(do|would)\s+you\s+(think|say)\s+i'?m"
    r"|how\s+can\s+you\s+tell"
    r"|what\s+makes\s+you\s+think"
    r")",
    re.IGNORECASE,
)


def detect_why_question(user_message: str) -> bool:
    """C2: Return True if the user is asking ARIA to explain its mood reading."""
    return bool(_WHY_RE.search(user_message or ""))


def build_why_explanation(neighbors: list, rule_based: bool) -> str:
    """C2: Build an honest spoken explanation of how ARIA arrived at its mood
    reading — using the ACTUAL nearest-neighbor samples, never invented ones.

    If rule_based=True: explains that no personal data exists yet.
    If neighbors provided: names real dates + moods from the training history.
    """
    if rule_based or not neighbors:
        return (
            "I don't have enough of your own data yet to compare this to, "
            "so I'm going on general voice patterns rather than your specific history — "
            "things like how fast you're speaking and the pitch of your voice."
        )

    parts = []
    for n in neighbors:
        ts = n.get("timestamp", "")
        mood = n.get("mood", "")
        if not ts or not mood:
            continue
        # Format: "July 23" or "yesterday" etc. — readable date from ISO timestamp.
        try:
            dt = datetime.fromisoformat(ts.replace(" ", "T").split("T")[0])
            today = datetime.now().date()
            delta = (today - dt.date()).days
            if delta == 0:
                date_str = "earlier today"
            elif delta == 1:
                date_str = "yesterday"
            elif delta < 7:
                date_str = dt.strftime("%A")  # "Monday"
            else:
                date_str = dt.strftime("%B %d").replace(" 0", " ")  # cross-platform day without leading zero

        except Exception:
            date_str = ts[:10]
        parts.append(f"{date_str} ({mood})")

    if not parts:
        return (
            "I matched this to a few moments in your history, "
            "but I can't read the exact dates right now."
        )

    example_str = "; ".join(parts)
    return (
        f"A few times recently when your voice sounded similar to this — "
        f"like {example_str} — that's what the pattern pointed toward. "
        "I'm matching how you sound right now to those past moments."
    )


# ----------------------------------------------------------------------
# C3 — Weekly digest trigger detection
# ----------------------------------------------------------------------

_DIGEST_RE = re.compile(
    r"\b("
    r"how\s+has\s+(this\s+)?week\s+(been|gone)"
    r"|weekly\s+(summary|recap|reflection|digest|review)"
    r"|what'?s\s+my\s+week\s+been\s+like"
    r"|how\s+(have\s+i|did\s+i)\s+(been|do|feel)\s+this\s+week"
    r"|give\s+me\s+(a\s+)?(weekly\s+)?(reflection|summary|recap|overview)"
    r"|how\s+was\s+my\s+week"
    r")",
    re.IGNORECASE,
)


def detect_digest_request(user_message: str) -> bool:
    """C3: Return True if the user is asking for their weekly reflection digest."""
    return bool(_DIGEST_RE.search(user_message or ""))


# ----------------------------------------------------------------------
# C4 — Mentioned concern detection
# ----------------------------------------------------------------------

# Patterns that signal a future-oriented concern the user has mentioned.
# Named group `concern` captures the thing they're worried about.
_CONCERN_PATTERNS = [
    re.compile(r"\bi\s+have\s+(?:a\s+)?(?P<concern>[\w\s]{3,40}?)\s+(?:tomorrow|today|tonight|this\s+(?:week|weekend|afternoon|morning|evening)|on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|next\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\bi'?m\s+(?:a\s+bit\s+)?(?:worried|nervous|anxious|stressed)\s+about\s+(?P<concern>[\w\s]{3,60}?)(?:[.,!?]|$)", re.IGNORECASE),
    re.compile(r"\bhoping\s+(?:that\s+)?(?P<concern>[\w\s]{3,60}?)\s+(?:goes?|went|will\s+go)\s+well\b", re.IGNORECASE),
    re.compile(r"\b(?:got|have|there'?s)\s+(?:a\s+)?(?P<concern>[\w\s]{3,40}?)\s+coming\s+up\b", re.IGNORECASE),
    re.compile(r"\b(?P<concern>[\w\s]{3,40}?)\s+is\s+(?:tomorrow|today|tonight|this\s+(?:week|weekend|afternoon|morning|evening))\b", re.IGNORECASE),
]

# Time-window hints: how many hours to wait before following up.
_WINDOW_HINTS = {
    "tomorrow": 26, "today": 6, "tonight": 8,
    "this week": 72, "this weekend": 60, "next week": 120,
    "this morning": 4, "this afternoon": 4, "this evening": 6,
    "monday": 48, "tuesday": 48, "wednesday": 48,
    "thursday": 48, "friday": 48, "saturday": 48, "sunday": 48,
}


def detect_mentioned_concern(user_message: str) -> tuple:
    """C4: Detect a future-oriented concern in the user's message.

    Returns (concern_text: str | None, resolution_window_hours: int).
    concern_text is None when nothing is detected.
    window_hours defaults to 24h; extended for later timeframes.
    """
    lowered = (user_message or "").lower()
    for pattern in _CONCERN_PATTERNS:
        m = pattern.search(user_message)
        if m:
            try:
                concern = m.group("concern").strip().strip(".,!?")
            except IndexError:
                concern = user_message[:60]
            if len(concern) < 3:
                continue
            # Pick resolution window from time-hint keywords in the full message.
            window = 24
            for hint, hours in _WINDOW_HINTS.items():
                if hint in lowered:
                    window = hours
                    break
            return concern, window
    return None, 24


# Resolution phrases: user naturally references the topic in past tense /
# positive wrap-up — no need for ARIA to follow up.
_RESOLUTION_RE = re.compile(
    r"\b("
    r"it\s+went\s+(?:well|great|fine|okay|ok|really\s+well|pretty\s+well)"
    r"|all\s+(?:done|finished|good|sorted|over)"
    r"|it'?s\s+(?:done|finished|over|sorted|behind\s+me)"
    r"|went\s+(?:well|great|fine|okay)"
    r"|i\s+(?:did\s+it|nailed\s+it|passed|got\s+through\s+it|finished\s+it|survived\s+it)"
    r"|(?:turned|worked)\s+out\s+(?:well|great|fine|okay|alright)"
    r")",
    re.IGNORECASE,
)


def detect_concern_resolved(user_message: str, concern_text: str) -> bool:
    """C4: Return True if the user's message suggests a concern has been resolved.

    Two-part check:
    1. The concern topic is referenced in the current message (keyword overlap).
    2. The message contains a resolution phrase.
    Conservative — both must be true to avoid false positives.
    """
    if not user_message or not concern_text:
        return False
    # Simple keyword overlap: at least one non-trivial word from the concern
    # appears in the current message.
    concern_words = {
        w for w in concern_text.lower().split()
        if len(w) > 3 and w not in {"that", "this", "with", "about", "have", "been"}
    }
    msg_words = set(user_message.lower().split())
    topic_referenced = bool(concern_words & msg_words)
    return topic_referenced and bool(_RESOLUTION_RE.search(user_message))


def build_escalation_prompt(user_message: str, language: str = "en") -> str:
    """Dedicated, tightly-scoped prompt for the serious-distress path —
    deliberately separate from build_prompt() so this response can NEVER
    accidentally inherit the normal mood-response machinery (suggestions,
    YOUTUBE/SEARCH offers, proactive nudges). Explicitly forbids content
    suggestions of any kind."""
    language_name = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are ARIA, a personal AI assistant in a spoken voice conversation. The user just "
        "said something suggesting they may be going through real, serious emotional distress — "
        "not ordinary day-to-day stress.\n\n"
        f'They said: "{user_message}"\n\n'
        "RESPONSE INSTRUCTIONS — follow these exactly, they matter more than usual:\n"
        "- Acknowledge what they said genuinely and specifically, in a warm, calm, unhurried tone. "
        "No stock phrases like \"that sounds hard\" on its own.\n"
        "- Gently encourage them to reach out to someone they trust, or a mental health "
        "professional — be honest that this is beyond what you, as an AI, can really help with.\n"
        "- Do NOT suggest a video, a distraction, a search, a break, or content of any kind. "
        "This is not the moment for that.\n"
        "- Do NOT diagnose, minimize, or promise things will be fine — you don't know that.\n"
        "- Keep it short: 2-3 sentences, spoken naturally, not a lecture or a list.\n"
        f"- Respond in {language_name}.\n"
        "- Do not include CONFIDENCE, TASK, URL, SEARCH, YOUTUBE, or ACTION lines of any kind — "
        "plain spoken text only."
    )


# ----------------------------------------------------------------------
# Mood fusion
# ----------------------------------------------------------------------


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

def get_silent_directives(user_text, current_telemetry, user_id):
    directives = []
    
    words = user_text.strip().split()
    if len(words) <= 3 and user_text.strip().lower() in ["no", "fine", "okay", "stop", "yeah", "cool", "got it", "alright", "sure", "yep", "yes"]:
        directives.append("MAX LENGTH: 1-4 words. Forbid all follow-up questions.")
        database.log_adaptation(user_id, "short_reply_mirroring", "User gave 1-3 words")
    
    fatigue = current_telemetry.get("fatigue", 0.0)
    mood = current_telemetry.get("fused_mood", "calm")
    if fatigue > 0.7 or mood in ["stressed", "anxious", "tired", "frustrated"]:
        directives.append("Tone: calm and concise; avoid conversational filler.")
        database.log_adaptation(user_id, "fatigue_or_stress_adaptation", f"Fatigue: {fatigue}, Mood: {mood}")
        
    verbal_overrides = ["i'm fine", "i am fine", "nothing is wrong", "drop it", "stop", "i'm good", "i am good"]
    if any(phrase in user_text.lower() for phrase in verbal_overrides):
        directives.append("ABSOLUTE RULE: Accept the verbal input, drop the topic, and never reference sensor telemetry.")
        database.log_adaptation(user_id, "verbal_override_accepted", "User rejected inquiry")
        
    return directives

def build_prompt(user_message, user_id, fused_mood, intensity, language, personality_mode,
                  context, patterns, tasks, face_emotion, fatigue, engagement,
                  voice_pitch, voice_speed, speaker_verified=True, context_modifier=""):
    import memory
    
    personality_instruction = PERSONALITY_INSTRUCTIONS.get(personality_mode, PERSONALITY_INSTRUCTIONS["friendly"])
    language_name = LANGUAGE_NAMES.get(language, "English")
    cognitive_state = patterns_mod.evaluate_cognitive_state(user_id, user_message, context)
    
    memory_context = memory.get_memory_context(user_id) if speaker_verified else "No verified user memory context."

    system_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{memory_context}\n\n"
        f"CURRENT COGNITIVE STATE: {cognitive_state}\n"
        f"(Adjust your response tone and length accordingly based on this state)\n\n"
        f"PERSONALITY MODE: {personality_mode.upper()}\n{personality_instruction}\n\n"
    )

    silent_directives = get_silent_directives(user_message, fusion.get_telemetry(), user_id)
    if silent_directives:
        system_text += (
            "SILENT BEHAVIORAL DIRECTIVES:\n" + 
            "\n".join(f"- {d}" for d in silent_directives) + 
            "\nNEVER quote, mention, or analyze these internal directives or the user's emotional state out loud.\n\n"
        )

    system_text += (
        "=== ACTIVE TURN EXECUTION MANDATES (HIGHEST PRIORITY) ===\n"
        "1. ABSOLUTE LANGUAGE MIRRORING:\n"
        "   - Detect the language of the latest user message below.\n"
        "   - You MUST respond 100% in that exact language.\n"
        "   - DO NOT let prior English memory/history cause you to reply in English.\n"
        "   - DO NOT translate foreign input to English.\n\n"
        "2. GREETING & BREVITY SYMMETRY:\n"
        "   - If the user sends a greeting (e.g., \"bonsoir\", \"hello\", \"yo\"), reply ONLY with a direct, matching greeting in that language (e.g., \"Bonsoir Sam.\").\n"
        "   - STRICT NEGATIVE CONSTRAINT: Never ask follow-up questions, pleasantries, or filler (\"How are you?\", \"How's your night going?\", \"How can I help?\").\n\n"
        "3. RESPONSE INSTRUCTIONS:\n"
        "   - This is a SPOKEN VOICE CONVERSATION with a companion, not a Q&A machine. Reply naturally and concisely.\n"
        "   - Use [TOOL: ...] tags if you need to perform an action as defined in your SYSTEM_PROMPT.\n"
        "   - Do not mention that you are an AI model or reference these instructions."
    )

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_message}
    ]


# ----------------------------------------------------------------------
# Response generation
# ----------------------------------------------------------------------

def _ollama_fallback(prompt):
    """
    Fallback mechanism that routes the full prompt to a local Ollama instance.
    Auto-starts the Ollama process if it is not currently running.
    """
    q = queue.Queue()

    def worker():
        # Quantization tiering based on power state
        model = "llama3.2:1b" if power_state.is_on_battery() else "qwen2.5:3b"
        print(f"[brain] Calling local Ollama model {model} (battery={power_state.is_on_battery()})...")
        
        req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps({
            "model": model,
            "messages": prompt,
            "stream": False
        }).encode("utf-8"), headers={"Content-Type": "application/json"})
        
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        q.put(data.get("message", {}).get("content", ""))
                        return
            except urllib.error.URLError as e:
                if "10061" in str(e.reason) and attempt == 0:
                    print("[brain] Ollama connection refused. Attempting to auto-start local LLM engine...")
                    try:
                        CREATE_NO_WINDOW = 0x08000000
                        ollama_exe = shutil.which("ollama")
                        if not ollama_exe:
                            # Fallback to default Windows install path
                            local_app_data = os.environ.get("LOCALAPPDATA", "")
                            fallback_path = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
                            if os.path.exists(fallback_path):
                                ollama_exe = fallback_path
                                
                        if ollama_exe:
                            subprocess.Popen([ollama_exe, "serve"], creationflags=CREATE_NO_WINDOW)
                            time.sleep(3.0)
                            continue
                        else:
                            print("[brain] 'ollama' executable not found in PATH or default install location.")
                    except Exception as launch_e:
                        print(f"[brain] Failed to launch Ollama: {launch_e}")
                print(f"[brain] Local LLM fallback failed: {e}")
                break
            except Exception as e:
                print(f"[brain] Local LLM fallback error: {e}")
                break
                
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return q.get()


def get_ai_response(prompt):
    """Call Groq for the main conversational response. Retries on rate limits, falls back to Ollama on failure."""
    if client is not None:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=prompt,
                    # Room for long mathematical derivations and complete code blocks
                    # without truncating them. Length is shaped by the prompt rules.
                    max_tokens=2048,
                    temperature=0.7,
                )
                return completion.choices[0].message.content.strip()
            except RateLimitError:
                wait = 2 ** attempt
                print(f"[brain] Rate limited by Groq, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"[brain] get_ai_response failed with Groq: {e}. Attempting Ollama fallback...")
                break
    else:
        print("[brain] Groq client not configured. Falling back to local Ollama...")

    # Fallback path if Groq is unavailable or failed
    ollama_resp = _ollama_fallback(prompt)
    if ollama_resp:
        # We enforce the "CONFIDENCE:" requirement manually if the local model didn't output it
        if "CONFIDENCE:" not in ollama_resp:
            ollama_resp += "\nCONFIDENCE: Medium"
        return ollama_resp.strip()

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
{face_text}
{fatigue_text}

Respond in ONE short, natural sentence. Maximum 15 words.
Make it sound like a high-signal, professional AI assistant (e.g., 'Good {time_of_day} {name}, ready when you are.' or 'Good {time_of_day} {name}, what are we working on?').
Do NOT mention past tasks, past defenses, or previous conversations.
Do not offer help in a generic way like 'how can I assist you today?'
This is spoken aloud on app startup — brevity is critical."""

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
    # C2: global declaration MUST precede any use of these names in this scope.
    global _last_knn_neighbors, _last_prediction_was_rule_based

    clean_msg = (user_message or "").strip().lower()
    
    # Fast-path Tier 1: Instant response for short greeting calls (bypass LLM entirely)
    if clean_msg in DIRECT_CALLS:
        return {
            "ai_response": "Ready.",
            "text_mood": {"mood": "calm", "intensity": 0.5, "emotions": []},
            "prompt_mood": {"mood": "calm", "intensity": 0.5, "description": "User appears calm."}
        }

    voice_features = voice_features or {}
    face_features  = face_features or {}


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
    speaker_verified = face_features.get("speaker_verified", True)

    # Personal KNN classifier: once trained on this user's own history, its
    # prediction replaces the generic rule-based voice mood — the moment the
    # system stops guessing from fixed thresholds and starts using what it
    # learned about *this* user's voice/face signature. Only applied when the
    # current speaker IS that registered user — the model is trained on their
    # specific voice physiology, so applying it to someone else's voice would
    # be a meaningless nearest-neighbour lookup, not just a privacy concern.
    if speaker_verified:
        knn_mood, knn_neighbors = patterns_mod.predict_mood_knn_with_neighbors({
            "pitch": voice_pitch, "speaking_speed": voice_speed,
            "fatigue": fatigue, "engagement": engagement,
        })
        # Store neighbors at module level so the C2 explain path can access them.
        # rule_based flag comes from the return: None mood means no KNN model trained.
        _last_knn_neighbors = knn_neighbors
        _last_prediction_was_rule_based = (knn_mood is None)
        if knn_mood and voice_pitch > 0:
            voice_mood = knn_mood
            try:
                database.log_adaptation(user_id, "knn_mood_used",
                                        f"Personal classifier read this turn as '{knn_mood}' "
                                        f"(pitch {voice_pitch:.0f}Hz, {voice_speed:.1f} w/s)")
            except Exception:
                pass

    # Preliminary fusion based on face + voice (text mood requires LLM call)
    preliminary_fusion = fusion.fuse_moods(voice_mood, face_mood, "calm", voice_pitch, fatigue, engagement)

    prompt = build_prompt(
        user_message, user_id, preliminary_fusion["mood"], preliminary_fusion["intensity"],
        language, personality_mode, context, patterns, tasks,
        face_emotion, fatigue, engagement, voice_pitch, voice_speed,
        speaker_verified=speaker_verified,
        context_modifier=preliminary_fusion.get("context_modifier", "")
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        text_mood_future = executor.submit(analyze_text_mood, user_message)
        response_future = executor.submit(get_ai_response, prompt)
        text_mood_result = text_mood_future.result()
        ai_response = response_future.result()

    # Fast-path Tier 3: Apply post-processing sanitizer to LLM response
    ai_response = sanitize_response(ai_response)

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


import threading
from actions import ACTION_REGISTRY

TOOL_PATTERN = re.compile(r'\[TOOL:\s*(\w+)\((?:query|target|command)=["\'](.*?)["\']\)\]')

def process_interaction(raw_llm_output: str, voice_engine=None):
    """
    1. Extracts all tool calls.
    2. Strips tags to produce clean conversational text.
    3. Asynchronously dispatches tools in background daemon threads.
    4. Asynchronously dispatches speech synthesis.
    5. Returns clean text and executed action tags immediately to the caller.
    """
    tool_calls = TOOL_PATTERN.findall(raw_llm_output)
    clean_text = TOOL_PATTERN.sub('', raw_llm_output).strip()
    executed_tools = []

    # 1. Asynchronous tool execution
    for tool_name, tool_arg in tool_calls:
        if tool_name in ACTION_REGISTRY:
            action_fn = ACTION_REGISTRY[tool_name]
            threading.Thread(target=action_fn, args=(tool_arg,), daemon=True).start()
            executed_tools.append({"tool": tool_name, "arg": tool_arg})

    # 2. Asynchronous TTS execution
    if clean_text and voice_engine:
        threading.Thread(target=voice_engine, args=(clean_text,), daemon=True).start()

    return clean_text, executed_tools

def clean_response(response):
    """Aggressively strip old legacy tokens and extra whitespace."""
    cleaned = re.sub(r"^\s*(CONFIDENCE|TASK|URL|SEARCH|ACTION|YOUTUBE).*?$", "", response, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"(CONFIDENCE|TASK|URL|SEARCH|ACTION|YOUTUBE):\s*\w+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    cleaned = re.sub(r"\*.*?\*", "", cleaned)
    cleaned = re.sub(r"<.*?>", "", cleaned)
    cleaned = re.sub(r"\b(Whoa|Ohh|Function|Ah|Hmm|Oh)\b", "", cleaned, flags=re.IGNORECASE)
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
