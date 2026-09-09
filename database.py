"""
ARIA Desktop - Database Layer
SQLite persistence for users, conversations, patterns, tasks, and mood readings.
"""

import sqlite3
import os
import threading
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aria.db")


def _now() -> str:
    """Single timestamp convention for every write: local time, ISO format.

    SQLite's CURRENT_TIMESTAMP defaults are UTC with a space separator, while
    Python writes used local isoformat — mixing the two broke string-ordered
    date comparisons (' ' < 'T') and shifted day boundaries. All inserts now
    pass this explicitly instead of relying on column defaults.
    """
    return datetime.now().isoformat(timespec="seconds")

# sqlite3 connections are not thread-safe across threads by default.
# ARIA runs voice, camera, and UI on separate threads, so we keep one
# connection per thread via threading.local().
_local = threading.local()


def get_connection():
    """Return a thread-local SQLite connection (creates one if needed)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def init_db():
    """Create all tables if they do not already exist. Safe to call every startup."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            personality_mode TEXT DEFAULT 'friendly',
            language_preference TEXT DEFAULT 'en',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_message TEXT,
            ai_response TEXT,
            mood TEXT,
            confidence TEXT,
            language TEXT,
            voice_pitch REAL,
            voice_speed REAL,
            face_emotion TEXT,
            fatigue_level REAL,
            engagement_level REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
            user_message,
            ai_response,
            content='conversations',
            content_rowid='id'
        )
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
            INSERT INTO conversations_fts(rowid, user_message, ai_response)
            VALUES (new.id, new.user_message, new.ai_response);
        END;
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
            INSERT INTO conversations_fts(conversations_fts, rowid, user_message, ai_response)
            VALUES('delete', old.id, old.user_message, old.ai_response);
        END;
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
            INSERT INTO conversations_fts(conversations_fts, rowid, user_message, ai_response)
            VALUES('delete', old.id, old.user_message, old.ai_response);
            INSERT INTO conversations_fts(rowid, user_message, ai_response)
            VALUES (new.id, new.user_message, new.ai_response);
        END;
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            pattern_type TEXT,
            pattern_value TEXT,
            frequency INTEGER DEFAULT 1,
            last_seen DATETIME,
            UNIQUE(user_id, pattern_type, pattern_value),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            task_description TEXT,
            due_date TEXT,
            is_completed INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            voice_mood TEXT,
            face_mood TEXT,
            text_mood TEXT,
            fused_mood TEXT,
            intensity REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # A single-row-per-user cache of the computed behavioural profile, kept
    # fresh by patterns.py after every turn - lets the UI (and brain.py's
    # prompt building) read "what ARIA has learned" without recomputing it
    # from raw conversations/patterns/mood_readings on every access.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS behavioural_profile (
            user_id INTEGER PRIMARY KEY,
            dominant_mood TEXT,
            mood_trend TEXT,
            active_hour TEXT,
            frequent_topic TEXT,
            language_used TEXT,
            face_emotion_pattern TEXT,
            summary TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Chronological log of moments where learned data actually changed ARIA's
    # behaviour (KNN prediction used, proactive suggestion injected, pattern
    # referenced in a greeting, ...). This is the evidence trail the Memory
    # panel shows to answer "how does ARIA actually learn about its user?".
    cur.execute("""
        CREATE TABLE IF NOT EXISTS adaptation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event TEXT,
            detail TEXT,
            timestamp DATETIME,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Cooldown tracking for proactive suggestions (patterns.get_proactive_suggestion) —
    # without this, the same pending task or topic nudge gets offered on
    # literally every turn since it's otherwise a pure function of
    # (pending_tasks, patterns, hour) with no memory of having just said it.
    # suggestion_key is a stable identity for what was suggested (e.g.
    # "task:<id>" or "topic:<value>"), not the rendered text, so cooldown
    # survives minor wording changes.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proactive_suggestion_state (
            user_id INTEGER,
            suggestion_key TEXT,
            last_surfaced DATETIME,
            PRIMARY KEY (user_id, suggestion_key)
        )
    """)

    # Every other table's lookups filter by user_id - index it everywhere
    # it's queried so those filters don't degenerate into full table scans
    # as conversation/pattern/mood history grows across a long-running app.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_user ON patterns(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mood_readings_user ON mood_readings(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_adaptation_log_user ON adaptation_log(user_id)")

    # Ground-truth mood labels for evaluation (nullable — only set when the
    # user self-reports via "label <mood>" or the periodic check-in).
    # Lives on mood_readings because that row already carries the per-modality
    # readings (voice/face/text/fused) that accuracy is computed against.
    try:
        cur.execute("ALTER TABLE mood_readings ADD COLUMN ground_truth_mood TEXT")
        print("[database] migrated: mood_readings.ground_truth_mood added")
    except sqlite3.OperationalError:
        pass  # column already exists

    # C1: correction_source distinguishes user-correction ground-truth labels
    # ("user_correction") from self-report labels ("label" command) so the
    # KNN training pipeline can weight them appropriately and the adaptation
    # log can surface them separately.
    try:
        cur.execute("ALTER TABLE mood_readings ADD COLUMN correction_source TEXT")
        print("[database] migrated: mood_readings.correction_source added")
    except sqlite3.OperationalError:
        pass  # column already exists

    # C4: mentioned_concerns — stores future-oriented statements the user made
    # ("I have a presentation tomorrow") so ARIA can ask once after the window
    # passes if the topic wasn't naturally resolved in conversation.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mentioned_concerns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            concern_text TEXT,
            mentioned_at DATETIME,
            resolution_window_hours INTEGER DEFAULT 24,
            followed_up INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concerns_user ON mentioned_concerns(user_id)")

    conn.commit()
    print("[database] Database initialised at", DB_PATH)


class transaction:
    """
    Context manager for grouping multiple related writes into a single
    commit/rollback - use when several statements must succeed or fail
    together (eg. updating several behavioural patterns from one message).
    Single-statement functions elsewhere already commit individually, which
    is sufficient for them; this is for the multi-statement case.
    """

    def __init__(self):
        self.conn = get_connection()

    def __enter__(self):
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False


# ----------------------------------------------------------------------
# Behavioural profile
# ----------------------------------------------------------------------

def get_behavioural_profile(user_id):
    """Return the cached behavioural profile for a user as a dict, or None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM behavioural_profile WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def upsert_behavioural_profile(user_id, dominant_mood=None, mood_trend=None,
                                active_hour=None, frequent_topic=None,
                                language_used=None, face_emotion_pattern=None,
                                summary=None):
    """
    Insert or replace the behavioural profile row for a user.
    Only fields explicitly passed (not None) are written; existing values for
    omitted fields are preserved via read-modify-write so callers can update
    a single field without clobbering the rest.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM behavioural_profile WHERE user_id = ?", (user_id,))
    existing = cur.fetchone()
    existing = dict(existing) if existing else {}

    merged = {
        "dominant_mood":        dominant_mood        if dominant_mood        is not None else existing.get("dominant_mood"),
        "mood_trend":           mood_trend           if mood_trend           is not None else existing.get("mood_trend"),
        "active_hour":          active_hour          if active_hour          is not None else existing.get("active_hour"),
        "frequent_topic":       frequent_topic       if frequent_topic       is not None else existing.get("frequent_topic"),
        "language_used":        language_used        if language_used        is not None else existing.get("language_used"),
        "face_emotion_pattern": face_emotion_pattern if face_emotion_pattern is not None else existing.get("face_emotion_pattern"),
        "summary":              summary              if summary              is not None else existing.get("summary"),
    }

    cur.execute("""
        INSERT INTO behavioural_profile
            (user_id, dominant_mood, mood_trend, active_hour, frequent_topic,
             language_used, face_emotion_pattern, summary, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            dominant_mood        = excluded.dominant_mood,
            mood_trend           = excluded.mood_trend,
            active_hour          = excluded.active_hour,
            frequent_topic       = excluded.frequent_topic,
            language_used        = excluded.language_used,
            face_emotion_pattern = excluded.face_emotion_pattern,
            summary              = excluded.summary,
            updated_at           = excluded.updated_at
    """, (
        user_id,
        merged["dominant_mood"],
        merged["mood_trend"],
        merged["active_hour"],
        merged["frequent_topic"],
        merged["language_used"],
        merged["face_emotion_pattern"],
        merged["summary"],
        datetime.now().isoformat(timespec="seconds"),
    ))
    conn.commit()


# ----------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------

def get_or_create_user(name):
    """Fetch the user by name (case-insensitive), or create a new one. Returns user_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE LOWER(name) = LOWER(?)", (name,))
    row = cur.fetchone()
    if row:
        user_id = row["id"]
        update_user_last_seen(user_id)
        return user_id

    cur.execute(
        "INSERT INTO users (name, last_seen) VALUES (?, ?)",
        (name, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    print(f"[database] Created new user '{name}' with id {cur.lastrowid}")
    return cur.lastrowid


def get_user(user_id):
    """Return the full user row as a dict, or None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_first_user():
    """Return the earliest-created user, or None if no users exist yet (first run)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    return dict(row) if row else None


def update_user_last_seen(user_id):
    """Stamp last_seen with the current time."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET last_seen = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="seconds"), user_id),
    )
    conn.commit()


def set_personality_mode(user_id, mode):
    """Persist the chosen personality mode (friendly/professional/motivational)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET personality_mode = ? WHERE id = ?", (mode, user_id))
    conn.commit()


def set_language_preference(user_id, language):
    """Persist the most recently detected/used language as the user's preference."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET language_preference = ? WHERE id = ?", (language, user_id))
    conn.commit()


# ----------------------------------------------------------------------
# Conversations
# ----------------------------------------------------------------------

def save_conversation(user_id, user_message, ai_response, mood, confidence, language,
                       voice_pitch, voice_speed, face_emotion, fatigue_level, engagement_level):
    """Persist one full conversational turn with all multimodal signals attached."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (
            user_id, user_message, ai_response, mood, confidence, language,
            voice_pitch, voice_speed, face_emotion, fatigue_level, engagement_level,
            timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, user_message, ai_response, mood, confidence, language,
        voice_pitch, voice_speed, face_emotion, fatigue_level, engagement_level,
        _now(),
    ))
    conn.commit()
    return cur.lastrowid


def get_conversation_history(user_id, limit=3):
    """Return the most recent `limit` turns, oldest first, as a list of dicts."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM conversations
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    rows.reverse()
    return rows


def get_all_conversations(user_id):
    """Return every conversation for a user, oldest first. Used for summarisation."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM conversations WHERE user_id = ? ORDER BY timestamp ASC
    """, (user_id,))
    return [dict(r) for r in cur.fetchall()]


def get_total_conversations(user_id):
    """Return the total number of conversational turns recorded for a user."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM conversations WHERE user_id = ?", (user_id,))
    return cur.fetchone()["c"]


# ----------------------------------------------------------------------
# Mood readings
# ----------------------------------------------------------------------

def save_mood_reading(user_id, voice_mood, face_mood, text_mood, fused_mood, intensity):
    """Persist one multimodal mood fusion snapshot."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO mood_readings (user_id, voice_mood, face_mood, text_mood, fused_mood, intensity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, voice_mood, face_mood, text_mood, fused_mood, intensity, _now()))
    conn.commit()
    return cur.lastrowid


def set_ground_truth_mood(user_id, mood, source=None):
    """Attach a ground-truth label to the user's most recent mood reading.

    source: optional string tag written to correction_source — e.g.
    "user_correction" (C1 automatic detection) vs None ("label" command).
    Returns the row id or None.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id FROM mood_readings WHERE user_id = ? ORDER BY id DESC LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    if not row:
        return None
    cur.execute(
        "UPDATE mood_readings SET ground_truth_mood = ?, correction_source = ? WHERE id = ?",
        (mood.strip().lower(), source, row["id"]),
    )
    conn.commit()
    return row["id"]


def get_labeled_mood_readings(user_id):
    """All readings that have a ground-truth label, oldest first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM mood_readings
        WHERE user_id = ? AND ground_truth_mood IS NOT NULL
        ORDER BY id ASC
    """, (user_id,))
    return [dict(r) for r in cur.fetchall()]


def get_mood_history(user_id, days=7):
    """Return mood_readings rows from the last `days` days, oldest first."""
    conn = get_connection()
    cur = conn.cursor()
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    cur.execute("""
        SELECT * FROM mood_readings
        WHERE user_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    """, (user_id, since))
    return [dict(r) for r in cur.fetchall()]


def get_all_mood_readings(user_id):
    """Return every mood reading for a user. Used to train the KNN classifier."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM mood_readings WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
    return [dict(r) for r in cur.fetchall()]


def get_recent_fused_moods(user_id, limit=5):
    """Most recent fused_mood values, newest first — used to detect a
    SUSTAINED mood pattern (several real turns in a row), as opposed to a
    single momentary reading (see patterns.is_mood_sustained)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT fused_mood FROM mood_readings
        WHERE user_id = ? ORDER BY id DESC LIMIT ?
    """, (user_id, limit))
    return [r["fused_mood"] for r in cur.fetchall()]


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

def update_pattern(user_id, pattern_type, value):
    """
    Upsert a behavioural pattern. If (user_id, pattern_type, value) already
    exists, bump its frequency and refresh last_seen; otherwise insert it fresh.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO patterns (user_id, pattern_type, pattern_value, frequency, last_seen)
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(user_id, pattern_type, pattern_value)
        DO UPDATE SET frequency = frequency + 1, last_seen = excluded.last_seen
    """, (user_id, pattern_type, value, now))
    conn.commit()


def get_patterns(user_id, limit=5):
    """Return the top `limit` patterns by frequency for a user."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM patterns
        WHERE user_id = ?
        ORDER BY frequency DESC, last_seen DESC
        LIMIT ?
    """, (user_id, limit))
    return [dict(r) for r in cur.fetchall()]


def get_patterns_by_type(user_id, pattern_type, limit=5):
    """Return the top patterns of a specific type (eg. 'active_hour')."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM patterns
        WHERE user_id = ? AND pattern_type = ?
        ORDER BY frequency DESC, last_seen DESC
        LIMIT ?
    """, (user_id, pattern_type, limit))
    return [dict(r) for r in cur.fetchall()]


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

def save_task(user_id, description, due_date):
    """Insert a new pending task."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tasks (user_id, task_description, due_date, created_at) VALUES (?, ?, ?, ?)
    """, (user_id, description, due_date, _now()))
    conn.commit()
    return cur.lastrowid


def get_pending_tasks(user_id):
    """Return all incomplete tasks for a user, most recently created first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM tasks
        WHERE user_id = ? AND is_completed = 0
        ORDER BY created_at DESC
    """, (user_id,))
    return [dict(r) for r in cur.fetchall()]


def get_all_tasks(user_id):
    """Return every task for a user (completed and pending), most recent first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    return [dict(r) for r in cur.fetchall()]


def complete_task(task_id):
    """Mark a task as completed."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE tasks SET is_completed = 1 WHERE id = ?", (task_id,))
    conn.commit()


# ----------------------------------------------------------------------
# Adaptation ledger
# ----------------------------------------------------------------------

def log_adaptation(user_id, event, detail=""):
    """Record one moment where learned data influenced ARIA's behaviour."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO adaptation_log (user_id, event, detail, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_id, event, detail, _now()))
    conn.commit()


def get_adaptation_log(user_id, limit=12):
    """Return the most recent adaptation events, newest first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM adaptation_log
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    return [dict(r) for r in cur.fetchall()]


# ----------------------------------------------------------------------
# Proactive suggestion cooldown
# ----------------------------------------------------------------------

def get_suggestion_last_surfaced(user_id, suggestion_key):
    """Return the ISO timestamp this suggestion_key was last surfaced to the
    user, or None if never (or not since it was last dismissed/changed)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT last_surfaced FROM proactive_suggestion_state
        WHERE user_id = ? AND suggestion_key = ?
    """, (user_id, suggestion_key))
    row = cur.fetchone()
    return row["last_surfaced"] if row else None


def mark_suggestion_surfaced(user_id, suggestion_key):
    """Record that this suggestion was just offered, starting its cooldown."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO proactive_suggestion_state (user_id, suggestion_key, last_surfaced)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, suggestion_key) DO UPDATE SET last_surfaced = excluded.last_surfaced
    """, (user_id, suggestion_key, _now()))
    conn.commit()


# ----------------------------------------------------------------------
# C4 — Mentioned concerns (proactive follow-through)
# ----------------------------------------------------------------------

def save_mentioned_concern(user_id, concern_text, resolution_window_hours=24):
    """Store a future-oriented concern the user just mentioned.

    resolution_window_hours: how long to wait before ARIA may follow up.
    After this window, if the concern hasn't been naturally resolved or
    followed-up on, get_pending_concerns() will surface it.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO mentioned_concerns
            (user_id, concern_text, mentioned_at, resolution_window_hours)
        VALUES (?, ?, ?, ?)
    """, (user_id, concern_text, _now(), resolution_window_hours))
    conn.commit()
    return cur.lastrowid


def get_pending_concerns(user_id):
    """Return concerns whose resolution window has passed and haven't been
    followed up or resolved yet. Oldest first.

    The window check is done via SQLite datetime arithmetic so it respects
    whatever local timezone was used when mentioned_at was written (same
    _now() convention as every other timestamp in this module).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM mentioned_concerns
        WHERE user_id = ?
          AND followed_up = 0
          AND resolved = 0
          AND datetime(mentioned_at, '+' || resolution_window_hours || ' hours') <= datetime('now', 'localtime')
        ORDER BY mentioned_at ASC
    """, (user_id,))
    return [dict(r) for r in cur.fetchall()]


def get_all_concerns(user_id, limit=20):
    """Return the most recent concerns for a user (all states), newest first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM mentioned_concerns
        WHERE user_id = ?
        ORDER BY mentioned_at DESC
        LIMIT ?
    """, (user_id, limit))
    return [dict(r) for r in cur.fetchall()]


def mark_concern_followed_up(concern_id):
    """Record that ARIA asked the follow-up question for this concern.
    Once followed up, it will never be asked again (followed_up=1 gate).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE mentioned_concerns SET followed_up = 1 WHERE id = ?", (concern_id,))
    conn.commit()


def mark_concern_resolved(concern_id):
    """Mark a concern as resolved (user referenced it themselves before the
    follow-up window, or explicitly said it went fine).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE mentioned_concerns SET resolved = 1 WHERE id = ?", (concern_id,))
    conn.commit()


def get_recent_concern_count(user_id, event_type="mood_corrected", days=7):
    """Count adaptation_log events of a given type in the last `days` days.
    Used by generate_weekly_digest to count C1 mood corrections as an
    improvement-signal proxy.
    """
    conn = get_connection()
    cur = conn.cursor()
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    cur.execute("""
        SELECT COUNT(*) AS c FROM adaptation_log
        WHERE user_id = ? AND event = ? AND timestamp >= ?
    """, (user_id, event_type, since))
    return cur.fetchone()["c"]


def search_conversations(user_id: int, query: str, limit: int = 5) -> list[dict]:
    """Offline FTS5 search across the user's historical conversations."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, c.user_message, c.ai_response, c.timestamp 
        FROM conversations_fts fts
        JOIN conversations c ON fts.rowid = c.id
        WHERE conversations_fts MATCH ? AND c.user_id = ?
        ORDER BY rank
        LIMIT ?
    """, (query, user_id, limit))
    return [dict(r) for r in cur.fetchall()]


if __name__ == "__main__":

    # Quick manual smoke test: python database.py
    init_db()
    uid = get_or_create_user("Test User")
    print("User id:", uid)
    save_conversation(uid, "Hello", "Hi there!", "calm", "High", "en",
                       150.0, 2.1, "happy", 0.1, 0.8)
    print("History:", get_conversation_history(uid))
    update_pattern(uid, "active_hour", "14")
    print("Patterns:", get_patterns(uid))
    save_task(uid, "Submit report", "2026-07-01")
    print("Pending tasks:", get_pending_tasks(uid))
    save_mood_reading(uid, "calm", "happy", "calm", "happy", 0.7)
    print("Mood history:", get_mood_history(uid))
    upsert_behavioural_profile(uid, dominant_mood="calm", summary="Seems focused in the afternoon.")
    print("Behavioural profile:", get_behavioural_profile(uid))
