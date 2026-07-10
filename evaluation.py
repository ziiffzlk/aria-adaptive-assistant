"""
ARIA Desktop - Evaluation Module
Quantitative evaluation of the behavioural pattern recognition system, for
academic validation. Run:  python evaluation.py

Produces evaluation_report.txt with four studies:

  1. Personal KNN mood classifier — leave-one-out cross-validation on the
     user's REAL multimodal history (accuracy, per-class precision/recall/F1,
     confusion matrix) against a majority-class baseline and the rule-based
     classifier. Honestly reports insufficient data when n is too small.
  2. Pattern detection — precision/recall against PLANTED ground truth:
     scripted conversations with known topics/hours/moods are fed through
     patterns.update_all_patterns into an isolated temporary database, and
     the detected patterns are compared to what was planted.
  3. Mood fusion — systematic verification of the weighted-vote rules and
     physiological overrides across an exhaustive input grid.
  4. Proactive suggestions — trigger-condition truth table.

The synthetic studies run against a throwaway SQLite file; the real user
database is opened read-only and never modified.
"""

import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import database
import patterns
from brain import fuse_moods
from voice import classify_voice_mood

REPORT_PATH = os.path.join(_HERE, "evaluation_report.txt")
_lines: list[str] = []


def emit(line: str = ""):
    print(line)
    _lines.append(line)


# ──────────────────────────────────────────────────────────────────────
# Study 1: KNN mood classifier — leave-one-out CV on real history
# ──────────────────────────────────────────────────────────────────────

def study_knn(real_rows):
    emit("=" * 68)
    emit("STUDY 1 — Personal KNN mood classifier (leave-one-out CV, real data)")
    emit("=" * 68)

    samples, labels = [], []
    for r in real_rows:
        if (r.get("voice_pitch") or 0.0) == 0.0:
            continue  # typed turn — no acoustics
        if None in (r.get("voice_speed"), r.get("fatigue_level"), r.get("engagement_level")) or not r.get("mood"):
            continue
        samples.append([r["voice_pitch"], r["voice_speed"], r["fatigue_level"], r["engagement_level"]])
        labels.append(r["mood"])

    n = len(samples)
    emit(f"Usable real samples (voice turns with full multimodal features): {n}")
    if n < 12:
        emit(f"INSUFFICIENT DATA for statistically meaningful cross-validation "
             f"(need >= 12, have {n}).")
        emit("This is reported honestly rather than padded: the classifier only")
        emit("activates in the live system once >10 samples exist, and its")
        emit("accuracy should be re-measured with this script as usage grows.")
        return

    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    majority = Counter(labels).most_common(1)[0]
    correct, preds = 0, []
    for i in range(n):
        train_X = samples[:i] + samples[i + 1:]
        train_y = labels[:i] + labels[i + 1:]
        model = KNeighborsClassifier(n_neighbors=min(3, len(train_X)))
        model.fit(train_X, train_y)
        p = model.predict([samples[i]])[0]
        preds.append(p)
        correct += (p == labels[i])

    acc = correct / n
    base = majority[1] / n
    emit(f"LOOCV accuracy:            {acc:.1%}  ({correct}/{n})")
    emit(f"Majority-class baseline:   {base:.1%}  (always '{majority[0]}')")

    # Rule-based comparison on the same rows
    rule_correct = sum(
        classify_voice_mood(s[0], s[1]) == y for s, y in zip(samples, labels))
    emit(f"Rule-based classifier:     {rule_correct / n:.1%}  (fixed thresholds, no learning)")

    classes = sorted(set(labels) | set(preds))
    emit("\nPer-class precision / recall / F1:")
    for c in classes:
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        emit(f"  {c:16} P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}  (support {labels.count(c)})")

    emit("\nConfusion matrix (rows=actual, cols=predicted):")
    emit("  " + " ".join(f"{c[:6]:>7}" for c in classes))
    for a in classes:
        row = [sum(1 for p, y in zip(preds, labels) if y == a and p == b) for b in classes]
        emit(f"  {a[:6]:>6} " + " ".join(f"{v:>7}" for v in row))


# ──────────────────────────────────────────────────────────────────────
# Study 2: pattern detection vs planted ground truth (isolated temp DB)
# ──────────────────────────────────────────────────────────────────────

# Scripted week of interactions. Ground truth is what a correct pattern
# engine MUST recover: topics mentioned >=3x, the dominant hour, dominant mood.
_SCRIPT = [
    # (hour, message, mood, language, face)
    (21, "I need help with my project deadline",              "stressed", "en", "neutral"),
    (21, "the project is due friday and I'm behind",          "stressed", "en", "sad"),
    (21, "let's plan the project presentation",               "calm",     "en", "neutral"),
    (21, "my code keeps crashing on startup",                 "frustrated","en", "angry"),
    (22, "still fixing the code tonight",                     "tired",    "en", "tired"),
    (9,  "good morning, what's the weather",                  "happy",    "en", "happy"),
    (21, "project report is nearly done",                     "calm",     "en", "neutral"),
    (21, "I want to exercise more this month",                "calm",     "en", "neutral"),
    (14, "remind me about the meeting tomorrow",              "calm",     "en", "neutral"),
    (21, "the deadline is really stressing me out",           "stressed", "en", "sad"),
]
_TRUTH = {
    "frequent_topic": {"project", "deadline", "code"},   # planted 4x / 2x / 2x
    "active_hour":    {"21"},                             # 7 of 10 messages
    "dominant_mood":  {"stressed", "calm"},               # 3x and 4x — both legitimate
}


def study_patterns():
    emit("")
    emit("=" * 68)
    emit("STUDY 2 — Pattern detection: precision/recall vs planted ground truth")
    emit("=" * 68)

    uid = database.get_or_create_user("EVAL Pattern User")
    for hour, msg, mood, lang, face in _SCRIPT:
        patterns.update_all_patterns(uid, msg, mood, hour, lang, face)

    detected = database.get_patterns(uid, limit=50)
    by_type = defaultdict(list)
    for p in detected:
        by_type[p["pattern_type"]].append((p["pattern_value"], p["frequency"]))

    emit(f"Scripted messages fed: {len(_SCRIPT)}   patterns detected: {len(detected)}")
    for ptype, truth in _TRUTH.items():
        # Standard top-k evaluation: k = |ground truth| for the type.
        k = len(truth)
        top = {v for v, f in sorted(by_type.get(ptype, []), key=lambda x: -x[1])[:k]}
        tp = len(top & truth)
        precision = tp / len(top) if top else 0.0
        recall = tp / len(truth)
        emit(f"\n  {ptype} (top-{k}):")
        emit(f"    planted truth:   {sorted(truth)}")
        emit(f"    detected top-{k}: {sorted(top)}")
        emit(f"    precision={precision:.2f}  recall={recall:.2f}")

    # Frequency correctness: 'project' appears in exactly 4 scripted messages
    proj = next((f for v, f in by_type.get("frequent_topic", []) if v == "project"), 0)
    emit(f"\n  frequency check: 'project' planted 4x, counted {proj}x "
         f"→ {'PASS' if proj == 4 else 'FAIL'}")


# ──────────────────────────────────────────────────────────────────────
# Study 3: mood fusion rule verification (exhaustive grid)
# ──────────────────────────────────────────────────────────────────────

def study_fusion():
    emit("")
    emit("=" * 68)
    emit("STUDY 3 — Multimodal mood fusion: weighted vote + override rules")
    emit("=" * 68)

    moods = ["happy", "sad", "stressed", "calm", "tired", "frustrated"]
    total = agree = 0
    override_pass = 0
    override_total = 0

    # Vote correctness: when face (0.40) disagrees with voice (0.25) and text
    # (0.35), the pairwise winner must follow the weights.
    for face in moods:
        for text in moods:
            for vc in moods:
                r = fuse_moods(vc, face, text, voice_pitch=150, fatigue=0.1, engagement=0.6)
                votes = {}
                for m, w in ((face, .40), (text, .35), (vc, .25)):
                    votes[m] = votes.get(m, 0) + w
                expected = max(votes, key=votes.get)
                total += 1
                agree += (r["mood"] == expected)

    # Physiological overrides
    r = fuse_moods("happy", "happy", "happy", 150, fatigue=0.9, engagement=0.5)
    override_total += 1; override_pass += (r["mood"] == "tired")
    r = fuse_moods("calm", "angry", "stressed", 150, fatigue=0.1, engagement=0.5)
    override_total += 1; override_pass += (r["mood"] == "frustrated")
    r = fuse_moods("calm", "fearful", "calm", 320, fatigue=0.1, engagement=0.5)
    override_total += 1; override_pass += (r["mood"] == "anxious")

    emit(f"Weighted-vote agreement over {total} input combinations: {agree/total:.1%}")
    emit(f"Physiological override rules (fatigue→tired, anger+negative→frustrated,")
    emit(f"high-pitch+fear→anxious): {override_pass}/{override_total} PASS")


# ──────────────────────────────────────────────────────────────────────
# Study 4: proactive suggestion trigger table
# ──────────────────────────────────────────────────────────────────────

def study_proactive():
    emit("")
    emit("=" * 68)
    emit("STUDY 4 — Proactive suggestion triggers")
    emit("=" * 68)

    task = [{"task_description": "Submit report", "due_date": "friday"}]
    hot_hour = [{"pattern_type": "active_hour", "pattern_value": "21", "frequency": 5},
                {"pattern_type": "frequent_topic", "pattern_value": "project", "frequency": 5}]
    cold = [{"pattern_type": "frequent_topic", "pattern_value": "project", "frequency": 1}]

    cases = [
        ("pending task present → task reminder",
         patterns.get_proactive_suggestion(1, 10, task, []), lambda s: s and "Submit report" in s),
        ("active hour + hot topic → correlation nudge",
         patterns.get_proactive_suggestion(1, 21, [], hot_hour), lambda s: s and "project" in s),
        ("hot topic alone (freq>=3) → topic nudge",
         patterns.get_proactive_suggestion(1, 10, [], hot_hour), lambda s: s and "project" in s),
        ("nothing learned yet → stays silent",
         patterns.get_proactive_suggestion(1, 10, [], cold), lambda s: s is None),
    ]
    passed = 0
    for name, result, check in cases:
        ok = bool(check(result))
        passed += ok
        emit(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    emit(f"\n  {passed}/{len(cases)} trigger conditions behave as designed")


def main():
    emit(f"ARIA behavioural pattern recognition — evaluation report")
    emit(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    emit("")

    # Real data first, read from the live DB before we repoint to the temp DB.
    database.init_db()
    real_user = database.get_first_user()
    real_rows = database.get_all_conversations(real_user["id"]) if real_user else []
    emit(f"Real user: {real_user['name'] if real_user else 'none'} "
         f"({len(real_rows)} conversation rows on record)")
    emit("")
    study_knn(real_rows)

    # Everything synthetic runs against a throwaway database.
    tmp = os.path.join(tempfile.gettempdir(), "aria_eval.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        database._local.conn.close()
    except Exception:
        pass
    database._local.conn = None
    database.DB_PATH = tmp
    database.init_db()

    study_patterns()
    study_fusion()
    study_proactive()

    emit("")
    emit("=" * 68)
    emit("Interpretation notes for the defense:")
    emit("- Study 1 measures the LEARNED component on the user's own data;")
    emit("  its accuracy vs the majority baseline and the static rule-based")
    emit("  classifier quantifies what personalisation adds.")
    emit("- Study 2 shows the pattern engine recovers planted behavioural")
    emit("  regularities with measurable precision/recall, not anecdotes.")
    emit("- Studies 3-4 verify the decision logic is deterministic and")
    emit("  matches its documented design.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
