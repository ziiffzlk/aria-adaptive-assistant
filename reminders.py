"""
ARIA Desktop - Reminders Module
Background thread that watches pending tasks and fires a Windows toast
notification when a due date approaches — ARIA reminding proactively instead
of only mentioning tasks when asked.

Due dates are free text extracted by the LLM ("2026-07-01", "tomorrow 3pm",
"Friday"), so parsing is best-effort via dateutil; unparseable dates are
silently skipped rather than guessed.
"""

import threading
import time
from datetime import datetime, timedelta

from dateutil import parser as date_parser

import database

CHECK_INTERVAL_S = 60          # how often to scan pending tasks
NOTIFY_WINDOW_MIN = 30         # notify when a task is due within this many minutes

_stop = threading.Event()
_notified_ids: set[int] = set()   # session-scoped: one toast per task


def _parse_due(due: str):
    """Best-effort parse of a free-text due date. Returns (datetime, date_only) or (None, False)."""
    if not due or due.strip().lower() in ("unspecified", "none", "no deadline"):
        return None, False
    try:
        dt = date_parser.parse(due, fuzzy=True, default=datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0))
        date_only = ":" not in due and not any(w in due.lower() for w in ("am", "pm", "hour"))
        return dt, date_only
    except (ValueError, OverflowError):
        return None, False


def _toast(title: str, message: str):
    try:
        from plyer import notification
        notification.notify(title=title, message=message, app_name="ARIA", timeout=10)
        return True
    except Exception as e:
        print(f"[reminders] toast failed: {e}")
        return False


def _check_once(user_id):
    now = datetime.now()
    for task in database.get_pending_tasks(user_id):
        tid = task.get("id")
        if tid in _notified_ids:
            continue
        due_dt, date_only = _parse_due(task.get("due_date") or "")
        if due_dt is None:
            continue

        desc = (task.get("task_description") or "your task").strip()
        if date_only:
            # Date without a time: remind once on the morning of the day.
            if due_dt.date() == now.date():
                if _toast("ARIA — due today", f"{desc} is due today."):
                    _notified_ids.add(tid)
                    database.log_adaptation(user_id, "proactive_reminder",
                                            f"Toast reminder: '{desc[:80]}' due today")
        else:
            delta = due_dt - now
            if timedelta(0) <= delta <= timedelta(minutes=NOTIFY_WINDOW_MIN):
                mins = max(1, int(delta.total_seconds() // 60))
                if _toast("ARIA — coming up", f"{desc} — in about {mins} min."):
                    _notified_ids.add(tid)
                    database.log_adaptation(user_id, "proactive_reminder",
                                            f"Toast reminder: '{desc[:80]}' due in {mins} min")


def _loop(user_id):
    print("[reminders] Task reminder thread started")
    while not _stop.wait(CHECK_INTERVAL_S):
        try:
            _check_once(user_id)
        except Exception as e:
            print(f"[reminders] check failed: {e}")
    print("[reminders] Task reminder thread stopped")


def start(user_id):
    """Start the reminder watcher (daemon thread). Call once from main."""
    _stop.clear()
    threading.Thread(target=_loop, args=(user_id,), daemon=True).start()


def stop():
    _stop.set()


if __name__ == "__main__":
    # Quick manual smoke test: python reminders.py
    database.init_db()
    uid = database.get_or_create_user("Reminder Test User")
    in_10 = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M")
    database.save_task(uid, "Smoke-test reminder", in_10)
    _check_once(uid)
    print("Notified ids:", _notified_ids)
