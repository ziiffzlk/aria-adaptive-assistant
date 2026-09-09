"""
ARIA Desktop - System Actions Module
Safe, allowlisted desktop actions: open a known application, open a standard
user folder, or read basic system info. The LLM requests these via an
"ACTION: verb|target" line (parsed by brain.extract_action); nothing here
ever executes free-form input — every verb and target must match the
allowlists below or the action is refused with a helpful suggestion.

The app allowlist is built at import time: system apps that ship with Windows
are always included, and popular third-party apps (browsers, Office, Spotify,
VS Code, Windows Terminal) are included ONLY if actually detected on this
machine — ARIA never claims it can open something that isn't installed.
"""

import os
import shutil
import subprocess
from datetime import datetime

# ── allowlist construction (runs once at import) ────────────────────────

# Windows built-ins: resolvable via PATH on every standard install.
_SYSTEM_APPS: dict[str, list[str]] = {
    "notepad":       ["notepad.exe"],
    "calculator":    ["calc.exe"],
    "paint":         ["mspaint.exe"],
    "explorer":      ["explorer.exe"],
    "task_manager":  ["taskmgr.exe"],
    "control_panel": ["control.exe"],
    "powershell":    ["powershell.exe"],
    # "start" resolves protocol handlers; the argument is a fixed literal
    # from this dict, never user input, so cmd here is not shell injection.
    "settings":      ["cmd", "/c", "start", "", "ms-settings:"],
    "browser":       ["cmd", "/c", "start", "", "https://www.google.com"],  # default browser
}

# Popular apps: (candidate install paths, PATH fallback name). Included only
# when a real executable is found.
_DETECTABLE: dict[str, tuple[list[str], str | None]] = {
    "chrome": ([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ], "chrome"),
    "edge": ([
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ], "msedge"),
    "firefox": ([
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
    ], "firefox"),
    "word": ([
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
    ], None),
    "excel": ([
        r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE",
    ], None),
    "powerpoint": ([
        r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE",
    ], None),
    "spotify": ([r"%APPDATA%\Spotify\Spotify.exe"], "spotify"),
    "vscode":  ([r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
                 r"C:\Program Files\Microsoft VS Code\Code.exe"], "code"),
    "terminal": ([], "wt"),
}


def _detect_apps() -> dict[str, list[str]]:
    apps = dict(_SYSTEM_APPS)
    for name, (paths, which_name) in _DETECTABLE.items():
        exe = next((p for p in (os.path.expandvars(c) for c in paths)
                    if os.path.isfile(p)), None)
        if exe is None and which_name:
            exe = shutil.which(which_name)
        if exe:
            apps[name] = [exe]
    return apps


_APPS: dict[str, list[str]] = _detect_apps()
print(f"[system_actions] {len(_APPS)} apps available: {', '.join(sorted(_APPS))}")

# Allowlisted user folders, resolved relative to the user profile.
# appdata is the Roaming profile (where many apps keep their user config).
_FOLDERS: dict[str, str] = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop":   "Desktop",
    "pictures":  "Pictures",
    "music":     "Music",
    "videos":    "Videos",
    "home":      "",
    "appdata":   os.path.join("AppData", "Roaming"),
}

_INFO_TARGETS = {"time", "date", "battery"}

# A few friendly examples for refusal messages and prompt hints.
_EXAMPLE_APPS = [a for a in ("chrome", "spotify", "word", "notepad", "calculator")
                 if a in _APPS][:4]


def get_available_apps() -> list[str]:
    """Sorted app targets actually available on this machine (for prompts/UI)."""
    return sorted(_APPS)


def get_available_folders() -> list[str]:
    return sorted(_FOLDERS)


# How users actually refer to each app in speech — used to verify the LLM's
# chosen target was genuinely requested before anything launches.
_APP_ALIASES: dict[str, tuple[str, ...]] = {
    "vscode":        ("vscode", "vs code", "visual studio", "code"),
    "task_manager":  ("task manager", "taskmgr"),
    "control_panel": ("control panel",),
    "powershell":    ("powershell", "power shell", "shell"),
    "terminal":      ("terminal", "command line", "cmd"),
    "browser":       ("browser", "internet", "the web"),
    "explorer":      ("explorer", "file explorer", "my files"),
    "calculator":    ("calculator", "calc"),
    "word":          ("word", "document"),
    "excel":         ("excel", "spreadsheet"),
    "powerpoint":    ("powerpoint", "power point", "presentation", "slides"),
    "spotify":       ("spotify", "music app"),
}


def match_app_from_text(text: str) -> str | None:
    """
    Best-effort LOCAL match of "open <app>" phrasing to a real, available
    app, using the same alias table as target_was_requested. Used by
    offline_router.py to trigger app-opening without Groq — deliberately
    narrow (requires the word "open") so it doesn't accidentally fire on
    unrelated sentences that happen to mention an app name in passing.
    Returns the app key, or None if nothing matched.
    """
    lowered = (text or "").lower()
    if "open" not in lowered:
        return None
    for app in _APPS:
        aliases = _APP_ALIASES.get(app, ()) or (app.replace("_", " "),)
        if app not in _APP_ALIASES:
            aliases = (app.replace("_", " "), app)
        if any(a in lowered for a in aliases):
            return app
    return None


def target_was_requested(verb: str, target: str, user_message: str) -> bool:
    """
    Deterministic guard against LLM substitution: when refusing an app it can't
    open, the model sometimes ALSO emits an ACTION for a different app it
    suggested ("I can open Firefox instead" + ACTION: open_app|firefox).
    An open_app/open_folder action only executes if the user's own message
    actually mentions the target (or a natural alias of it).
    """
    if verb == "system_info":
        return True  # harmless read-only info
    lowered = (user_message or "").lower()
    aliases = _APP_ALIASES.get(target, ()) or (target.replace("_", " "),)
    if target not in _APP_ALIASES:
        aliases = (target.replace("_", " "), target)
    return any(a in lowered for a in aliases)


# ── action execution ─────────────────────────────────────────────────────

def perform(verb: str, target: str) -> dict:
    """
    Execute one allowlisted action. Returns
    {"ok": bool, "spoken": str} — `spoken` is a short sentence ARIA can say.
    Unknown verbs/targets are refused with a helpful alternative, never guessed.
    """
    verb = (verb or "").strip().lower()
    target = (target or "").strip().lower()

    try:
        if verb == "open_app":
            return _open_app(target)
        if verb == "open_folder":
            return _open_folder(target)
        if verb == "system_info":
            return _system_info(target)
        return {"ok": False,
                "spoken": f"I don't have a safe way to do '{verb}' — I can open apps, "
                          "open folders, or check the time, date, or battery."}
    except Exception as e:
        print(f"[system_actions] {verb}|{target} failed: {e}")
        return {"ok": False, "spoken": "I tried, but that didn't work on this machine."}


def _open_app(target: str) -> dict:
    argv = _APPS.get(target)
    if argv is None:
        examples = ", ".join(_EXAMPLE_APPS)
        return {"ok": False,
                "spoken": f"I can only open a specific set of apps right now — try {examples}."}
    subprocess.Popen(argv, shell=False,
                     creationflags=subprocess.CREATE_NO_WINDOW if argv[0] == "cmd" else 0)
    print(f"[system_actions] opened app: {target}")
    return {"ok": True, "spoken": f"Opening {target.replace('_', ' ')} for you."}


def _open_folder(target: str) -> dict:
    sub = _FOLDERS.get(target)
    if sub is None:
        allowed = ", ".join(sorted(_FOLDERS))
        return {"ok": False, "spoken": f"I can open these folders: {allowed}."}
    path = os.path.join(os.path.expanduser("~"), sub) if sub else os.path.expanduser("~")
    if not os.path.isdir(path):
        return {"ok": False, "spoken": f"I couldn't find your {target} folder."}
    os.startfile(path)
    print(f"[system_actions] opened folder: {path}")
    return {"ok": True, "spoken": f"Opening your {target} folder."}


def _system_info(target: str) -> dict:
    if target not in _INFO_TARGETS:
        return {"ok": False, "spoken": "I can tell you the time, date, or battery level."}

    now = datetime.now()
    if target == "time":
        return {"ok": True, "spoken": f"It's {now.strftime('%I:%M %p').lstrip('0')}."}
    if target == "date":
        return {"ok": True, "spoken": f"Today is {now.strftime('%A, %B %d, %Y')}."}

    # battery
    try:
        import psutil
        batt = psutil.sensors_battery()
        if batt is None:
            return {"ok": True, "spoken": "This machine doesn't report a battery — probably a desktop."}
        state = "charging" if batt.power_plugged else "on battery"
        return {"ok": True, "spoken": f"Battery is at {int(batt.percent)} percent, {state}."}
    except Exception as e:
        print(f"[system_actions] battery read failed: {e}")
        return {"ok": False, "spoken": "I couldn't read the battery level."}


if __name__ == "__main__":
    # Quick manual smoke test: python system_actions.py
    print("Available apps:", get_available_apps())
    print("Available folders:", get_available_folders())
    print(perform("system_info", "time"))
    print(perform("system_info", "battery"))
    print(perform("open_app", "definitely_not_allowed"))
    print(perform("run_shell", "whatever"))
