"""
ARIA Desktop - Offline Intent Router
When network_state.is_online() is False, Groq is unreachable — but almost
everything ARIA does already runs entirely locally: task storage, learned
patterns/behavioural profile, and multimodal mood fusion (voice+face+text)
never touched the network in the first place, and system actions are
allowlisted local subprocess calls. This module answers as many message
categories as possible from local data, so the app degrades gracefully
instead of freezing on a Groq call that can never return.

try_offline_response(message, user_id, fused_mood, fused_intensity) returns
(response_text, category) if the message matched a locally-answerable
category, or (None, None) if it's genuine open-ended conversation — the one
real capability that needs the network. The caller (app_web.py) shows the
honest "I can't have an open conversation right now" fallback for that case.
"""
import random
import re
from datetime import datetime
import subprocess
import urllib.request
import urllib.error
import json
import time
import shutil
import glob
import os
import threading
import queue

import database
import patterns
import system_actions
import power_state

# ----------------------------------------------------------------------
# Category matchers
# ----------------------------------------------------------------------

_TASK_LIST_RE = re.compile(
    r"\bwhat (?:tasks|do i (?:still )?(?:need|have)(?: to do)?)\b|"
    r"\bmy (?:pending )?tasks\b|\bwhat.?s (?:left|on my plate|pending)\b|"
    r"\bany(?:thing)? (?:pending|to do)\b",
    re.IGNORECASE,
)
_TASK_COMPLETE_RE = re.compile(
    r"\b(?:mark|complete|finish(?:ed)?)\b.*\bdone\b|"
    r"\bi(?:'ve| have)? (?:finished|completed|done with)\b|"
    r"\bthat.?s done\b",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(
    r"\bwhat have you learned\b|\bwhat do you know about me\b|"
    r"\bwhat.?s my profile\b|\btell me about myself\b|"
    r"\bwhat (?:patterns|have you) (?:noticed|learned)\b",
    re.IGNORECASE,
)
_MOOD_CHECKIN_RE = re.compile(
    r"\bhow do i seem\b|\bhow am i doing\b|\bwhat.?s my mood\b|"
    r"\bhow do you think i.?m feeling\b|\bhow do i look\b|\bcheck.?in\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(r"\bwhat time is it\b|\bwhat.?s the time\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\bwhat.?s the date\b|\bwhat day is it\b|\bwhat is today\b", re.IGNORECASE)
_VOLUME_RE = re.compile(r"\b(?:mute|unmute)\b|\bvolume (?:up|down|higher|lower)\b|\bturn (?:it )?(?:up|down)\b", re.IGNORECASE)
_SEARCH_RE = re.compile(r"\b(?:what do you remember|search|find) (?:about )?(.+)", re.IGNORECASE)


def _is_task_complete_request(message):
    return bool(_TASK_COMPLETE_RE.search(message))


def _is_task_list_request(message):
    return bool(_TASK_LIST_RE.search(message))


def _is_profile_request(message):
    return bool(_PROFILE_RE.search(message))


def _is_mood_checkin_request(message):
    return bool(_MOOD_CHECKIN_RE.search(message))


def _is_time_request(message):
    return bool(_TIME_RE.search(message))


def _is_date_request(message):
    return bool(_DATE_RE.search(message))


def _is_volume_request(message):
    return bool(_VOLUME_RE.search(message))


def _is_search_request(message):
    return bool(_SEARCH_RE.search(message))


# ----------------------------------------------------------------------
# a. Task management — direct DB template, no LLM
# ----------------------------------------------------------------------

def _handle_task_list(user_id):
    tasks = database.get_pending_tasks(user_id)
    if not tasks:
        return "You don't have any pending tasks right now."
    if len(tasks) == 1:
        t = tasks[0]
        due = f", due {t['due_date']}" if t.get("due_date") else ""
        return f"You have one pending task: \"{t['task_description']}\"{due}."
    lines = []
    for t in tasks[:5]:
        due = f" (due {t['due_date']})" if t.get("due_date") else ""
        lines.append(f"\"{t['task_description']}\"{due}")
    more = f", and {len(tasks) - 5} more" if len(tasks) > 5 else ""
    return f"You have {len(tasks)} pending tasks: " + "; ".join(lines) + more + "."


def _handle_task_complete(message, user_id):
    tasks = database.get_pending_tasks(user_id)
    if not tasks:
        return "You don't have any pending tasks to mark done."
    lowered = message.lower()
    # Best-effort local match: keyword overlap between the message and each
    # task's description — no LLM available offline to do this more smartly.
    # Ambiguous or no match is answered honestly rather than guessed.
    scored = []
    for t in tasks:
        words = set(re.findall(r"\w+", t["task_description"].lower()))
        overlap = sum(1 for w in words if len(w) > 3 and w in lowered)
        if overlap:
            scored.append((overlap, t))
    if not scored:
        return ("I've got you saying that's done, but I'm not sure which task you mean while "
                "offline — I can't do the smart matching I'd normally do. Could you say the "
                "task a bit more specifically, or I can handle it once I'm back online?")
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return ("A couple of your tasks could match that — I don't want to guess wrong while "
                "offline. Which one did you mean?")
    task = scored[0][1]
    database.complete_task(task["id"])
    return f"Marked \"{task['task_description']}\" as done."


# ----------------------------------------------------------------------
# b. Behavioural profile — template from patterns/DB, no LLM
# ----------------------------------------------------------------------

def _handle_profile(user_id):
    top_patterns = database.get_patterns(user_id, limit=5)
    trend = patterns.detect_mood_trend(user_id)
    total = database.get_total_conversations(user_id)

    if not top_patterns and (trend.get("trend") == "insufficient_data"):
        return "I haven't learned enough about you yet to say much — the more we talk, the more I'll pick up."

    parts = [f"We've talked {total} times so far."]
    topics = [p["pattern_value"] for p in top_patterns if p["pattern_type"] == "frequent_topic"]
    if topics:
        parts.append(f"You tend to bring up {', '.join(topics[:3])} a lot.")
    hours = [p["pattern_value"] for p in top_patterns if p["pattern_type"] == "active_hour"]
    if hours:
        parts.append(f"You're usually most active around {hours[0]}:00.")
    if trend.get("dominant_mood"):
        parts.append(f"Lately you've mostly come across as {trend['dominant_mood']}, "
                     f"and that's been {trend['trend']}.")
    return " ".join(parts)


# ----------------------------------------------------------------------
# c. Mood check-in — template variations per mood, describing the ALREADY
# locally-fused signal (voice+face+text fusion never needed Groq at all)
# ----------------------------------------------------------------------

_MOOD_TEMPLATES = {
    "happy":       ["You seem to be in good spirits right now.",
                     "You're coming across pretty happy at the moment — good to see.",
                     "Looks like you're feeling good right now."],
    "excited":     ["You sound genuinely excited right now.",
                     "There's real energy in how you're coming across.",
                     "You seem pretty fired up at the moment."],
    "sad":         ["You seem a bit down right now.",
                     "You're coming across a little low at the moment.",
                     "Things seem a bit heavy for you right now."],
    "stressed":    ["You seem under some pressure right now.",
                     "You're coming across pretty stressed at the moment.",
                     "Things seem a bit much for you right now."],
    "anxious":     ["You seem a little uneasy right now.",
                     "There's some tension in how you're coming across.",
                     "You seem a bit on edge at the moment."],
    "tired":       ["You seem like you're running low on energy.",
                     "You're coming across pretty tired right now.",
                     "You seem like you could use some rest."],
    "frustrated":  ["You seem frustrated right now.",
                     "There's some real frustration in how you're coming across.",
                     "Something seems to be getting to you right now."],
    "confused":    ["You seem a bit thrown off right now.",
                     "You're coming across a little uncertain at the moment.",
                     "Something seems unclear to you right now."],
    "distracted":  ["You seem a little distracted right now.",
                     "Your attention seems to be elsewhere at the moment.",
                     "You seem a bit scattered right now."],
    "engaged":     ["You seem pretty engaged and focused right now.",
                     "You're coming across attentive and present at the moment.",
                     "You seem locked in right now."],
    "surprised":   ["You seem caught off guard right now.",
                     "Something seems to have surprised you.",
                     "You're coming across a bit startled at the moment."],
    "calm confident": ["You seem calm and steady right now.",
                        "You're coming across confident and settled at the moment.",
                        "You seem grounded right now."],
    "calm":        ["You seem pretty calm right now.",
                     "You're coming across settled and even at the moment.",
                     "You seem steady right now."],
}


def _handle_mood_checkin(fused_mood, fused_intensity):
    templates = _MOOD_TEMPLATES.get((fused_mood or "calm").lower(), _MOOD_TEMPLATES["calm"])
    line = random.choice(templates)
    if fused_intensity is not None and fused_intensity >= 0.75:
        line += " Pretty clearly, too, from how you sound and look."
    return line


# ----------------------------------------------------------------------
# d. System actions — local regex/alias matcher, bypasses Groq entirely
# ----------------------------------------------------------------------

def _handle_system_action(message):
    from brain import extract_action
    action = extract_action(message)
    if not action:
        return None
    verb, target = action
    result = system_actions.perform(verb, target)
    return result["spoken"]


def _handle_volume(message):
    msg = message.lower()
    if "mute" in msg:
        subprocess.run(["powershell", "-c", "(new-object -com wscript.shell).SendKeys([char]173)"], capture_output=True)
        return "Muted the system volume."
    elif "down" in msg or "lower" in msg:
        subprocess.run(["powershell", "-c", "1..5 | % { (new-object -com wscript.shell).SendKeys([char]174) }"], capture_output=True)
        return "Turned the volume down."
    else:
        subprocess.run(["powershell", "-c", "1..5 | % { (new-object -com wscript.shell).SendKeys([char]175) }"], capture_output=True)
        return "Turned the volume up."


def _handle_search(message, user_id):
    match = _SEARCH_RE.search(message)
    if not match:
        return None
    query = match.group(1).strip(" ?.")
    results = database.search_conversations(user_id, query)
    if not results:
        return f"I couldn't find anything in my local logs matching '{query}'."
    
    summary = f"I found {len(results)} matches. Most recent: "
    lines = [f'"{r["user_message"]}" -> "{r["ai_response"]}"' for r in results[:2]]
    return summary + "; ".join(lines)


# ----------------------------------------------------------------------
# e. Fallback — genuine open-ended conversation, the one real network need
# ----------------------------------------------------------------------

def _call_ollama_fallback(message, fused_mood, fused_intensity):
    q = queue.Queue()

    def worker():
        # Quantization tiering based on power state
        model = "llama3.2:1b" if power_state.is_on_battery() else "qwen2.5:3b"
        print(f"[offline_router] Calling local Ollama model {model} (battery={power_state.is_on_battery()})...")
        
        req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps({
            "model": model,
            "prompt": f"You are ARIA. Respond concisely to: {message}",
            "stream": False
        }).encode("utf-8"), headers={"Content-Type": "application/json"})
        
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        q.put(data.get("response", ""))
                        return
            except urllib.error.URLError as e:
                if "10061" in str(e.reason) and attempt == 0:
                    print("[offline_router] Ollama connection refused. Attempting to auto-start local LLM engine...")
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
                            print("[offline_router] 'ollama' executable not found in PATH or default install location.")
                    except Exception as launch_e:
                        print(f"[offline_router] Failed to launch Ollama: {launch_e}")
                print(f"[offline_router] Local LLM fallback failed: {e}")
                break
            except Exception as e:
                print(f"[offline_router] Local LLM fallback error: {e}")
                break
                
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return q.get()

_GGUF_MODEL = None

def _get_gguf_model():
    global _GGUF_MODEL
    if _GGUF_MODEL is not None:
        return _GGUF_MODEL
        
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[offline_router] llama-cpp-python not installed.")
        return None
        
    model_paths = glob.glob(os.path.join("models", "*.gguf"))
    if not model_paths:
        print("[offline_router] No GGUF models found in ./models directory.")
        return None
        
    # Prefer qwen or llama if multiple exist
    target_model = model_paths[0]
    for p in model_paths:
        if "qwen" in p.lower() or "llama" in p.lower():
            target_model = p
            break
            
    print(f"[offline_router] Lazy loading GGUF model: {target_model}...")
    try:
        _GGUF_MODEL = Llama(model_path=target_model, n_ctx=2048, verbose=False)
        return _GGUF_MODEL
    except Exception as e:
        print(f"[offline_router] Failed to load GGUF: {e}")
        return None

def _call_gguf_fallback(message):
    model = _get_gguf_model()
    if not model:
        return None
        
    q = queue.Queue()

    def worker():
        prompt = f"You are ARIA, a helpful AI assistant. Answer concisely.\nUser: {message}\nARIA:"
        print("[offline_router] Generating response with embedded GGUF model...")
        try:
            output = model(
                prompt,
                max_tokens=512,
                stop=["User:"],
                repeat_penalty=1.15,
                temperature=0.7,
                echo=False
            )
            q.put(output["choices"][0]["text"].strip())
        except Exception as e:
            print(f"[offline_router] GGUF generation failed: {e}")
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return q.get()

OFFLINE_FALLBACK_MESSAGE = (
    "I can't reach my main reasoning engine right now since I'm offline, and my local model isn't "
    "responding. But I can still tell you about your tasks, what I've learned about you, or check "
    "your mood."
)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def try_offline_response(message, user_id, fused_mood="calm", fused_intensity=0.5):
    """
    Attempt to answer `message` entirely locally. Returns (text, category)
    on a match, or (None, None) if this is genuine open-ended conversation
    (the caller should show OFFLINE_FALLBACK_MESSAGE in that case — this
    function deliberately does NOT return that fallback itself, so the
    caller can log the distinction between "matched a local category" and
    "genuinely couldn't help offline").
    """
    if not message:
        return None, None

    # System actions checked first — "open notepad" could otherwise false-
    # match nothing else, but checking it early keeps intent priority clear
    # and matches the instruction's own category ordering.
    action_response = _handle_system_action(message)
    if action_response:
        return action_response, "system_action"

    if _is_task_complete_request(message):
        return _handle_task_complete(message, user_id), "task_complete"

    if _is_task_list_request(message):
        return _handle_task_list(user_id), "task_list"

    if _is_profile_request(message):
        return _handle_profile(user_id), "behavioural_profile"

    if _is_mood_checkin_request(message):
        return _handle_mood_checkin(fused_mood, fused_intensity), "mood_checkin"

    if _is_time_request(message):
        return datetime.now().strftime("It is currently %I:%M %p."), "time"
        
    if _is_date_request(message):
        return datetime.now().strftime("Today is %A, %B %d, %Y."), "date"
        
    if _is_volume_request(message):
        return _handle_volume(message), "volume"
        
    if _is_search_request(message):
        return _handle_search(message, user_id), "search"

    # Attempt local LLM fallback for open-ended requests before giving up completely
    ollama_resp = _call_ollama_fallback(message, fused_mood, fused_intensity)
    if ollama_resp:
        return ollama_resp, "local_llm_fallback"
        
    # Secondary fallback to embedded GGUF
    gguf_resp = _call_gguf_fallback(message)
    if gguf_resp:
        return gguf_resp, "local_llm_fallback"

    return OFFLINE_FALLBACK_MESSAGE, "offline_fallback"
