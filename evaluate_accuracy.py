"""
ARIA Desktop - Mood accuracy & learning evaluation (academic defense pack)
Run:  python evaluate_accuracy.py

Reads the REAL database (read-only) and writes to ./evaluation/:
  - accuracy_report.txt      fused + per-modality accuracy vs ground truth,
                             classification report, honest sample-size notes
  - confusion_matrix.png     fused mood vs ground truth heatmap
  - learning_curve.png       prequential KNN accuracy per retrain cycle vs
                             the rule-based baseline
  - adaptation_summary.md    ledger totals by type + real timestamped examples
  - pattern_growth.png       cumulative patterns & conversations over time
"""

import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
EVAL_DIR = os.path.join(_HERE, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import database
from patterns import TOPIC_KEYWORDS
from voice import classify_voice_mood

# Validated chart pair (CVD-safe on light surfaces, checked 2026-07-03).
BLUE, RUST = "#3c78b8", "#b5674a"

_lines = []
def emit(s=""):
    print(s)
    _lines.append(s)


def study_accuracy(user_id):
    emit("=" * 66)
    emit("MOOD ACCURACY vs GROUND TRUTH (self-reported labels)")
    emit("=" * 66)
    rows = database.get_labeled_mood_readings(user_id)
    n = len(rows)
    emit(f"Labeled samples: {n}")
    if n == 0:
        emit("NO GROUND-TRUTH LABELS EXIST YET.")
        emit("Recommendation: collect at least 30 labels (>=3 per mood class you")
        emit("want to report) before presenting accuracy numbers in a defense.")
        emit("Labels accrue via the every-7th-turn check-in, or say 'label <mood>'")
        emit("after any exchange.")
        return
    if n < 10:
        emit(f"CAUTION: {n} labels is too few for stable statistics — numbers below")
        emit("are directional only. Recommend >=30 labels (>=3 per class).")

    gt = [r["ground_truth_mood"] for r in rows]
    modalities = {
        "fused": [r["fused_mood"] or "" for r in rows],
        "voice-only": [r["voice_mood"] or "" for r in rows],
        "face-only": [r["face_mood"] or "" for r in rows],
        "text-only": [r["text_mood"] or "" for r in rows],
    }
    emit("\nAccuracy vs the SAME ground truth (honest per-modality comparison):")
    for name, preds in modalities.items():
        acc = sum(p == t for p, t in zip(preds, gt)) / n
        emit(f"  {name:11} {acc:6.1%}")

    from sklearn.metrics import confusion_matrix, classification_report
    labels_sorted = sorted(set(gt) | set(modalities["fused"]))
    cm = confusion_matrix(gt, modalities["fused"], labels=labels_sorted)
    emit("\nConfusion matrix (fused vs ground truth) — rows=actual, cols=predicted:")
    emit("        " + " ".join(f"{c[:6]:>7}" for c in labels_sorted))
    for lab, row in zip(labels_sorted, cm):
        emit(f"  {lab[:6]:>6} " + " ".join(f"{v:>7}" for v in row))
    emit("\n" + classification_report(gt, modalities["fused"],
                                      labels=labels_sorted, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_sorted)), labels_sorted)
    for i in range(len(labels_sorted)):
        for j in range(len(labels_sorted)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#333")
    ax.set_xlabel("Predicted (fused)"); ax.set_ylabel("Ground truth (self-report)")
    ax.set_title(f"Fused mood vs ground truth (n={n})")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    emit(f"saved {path}")


def study_learning_curve(user_id):
    emit("")
    emit("=" * 66)
    emit("LEARNING CURVE — prequential KNN accuracy per retrain cycle")
    emit("=" * 66)
    rows = database.get_all_conversations(user_id)
    samples, labels = [], []
    for r in rows:
        if (r.get("voice_pitch") or 0.0) == 0.0 or not r.get("mood"):
            continue
        if None in (r.get("voice_speed"), r.get("fatigue_level"), r.get("engagement_level")):
            continue
        samples.append([r["voice_pitch"], r["voice_speed"], r["fatigue_level"], r["engagement_level"]])
        labels.append(r["mood"])
    n = len(samples)
    emit(f"Acoustic voice-turns available (chronological): {n}")
    if n < 8:
        emit(f"NOT ENOUGH DATA for a meaningful curve: have {n}, need >= 8 "
             f"(that's {8 - n} more spoken turns). Chart skipped honestly.")
        return

    from sklearn.neighbors import KNeighborsClassifier
    # Prequential: after every retrain cycle (5 samples), predict the NEXT
    # unseen sample with the model as it existed at that point in time.
    xs, knn_acc = [], []
    correct = tried = 0
    for i in range(5, n):
        model = KNeighborsClassifier(n_neighbors=min(3, i))
        model.fit(samples[:i], labels[:i])
        correct += (model.predict([samples[i]])[0] == labels[i])
        tried += 1
        if tried and (i - 4) % 1 == 0:
            xs.append(i)
            knn_acc.append(correct / tried)
    rule_acc = sum(classify_voice_mood(s[0], s[1]) == y for s, y in zip(samples, labels)) / n

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, [a * 100 for a in knn_acc], color=BLUE, lw=2,
            label="Personal KNN (cumulative prequential accuracy)")
    ax.axhline(rule_acc * 100, color=RUST, lw=2, ls="--",
               label=f"Static rule-based baseline ({rule_acc:.0%})")
    ax.set_xlabel("Training samples available (turns)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Mood classifier learning curve — {n} real voice turns")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(EVAL_DIR, "learning_curve.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    emit(f"Final prequential KNN accuracy: {knn_acc[-1]:.1%} over {tried} predictions")
    emit(f"Rule-based reference:           {rule_acc:.1%}")
    emit(f"saved {path}")


def study_adaptation_summary(user_id):
    emit("")
    emit("=" * 66)
    emit("ADAPTATION LEDGER SUMMARY")
    emit("=" * 66)
    conn = database.get_connection()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM adaptation_log WHERE user_id = ? ORDER BY id ASC", (user_id,))]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["event"]].append(r)

    md = ["# ARIA adaptation ledger summary", "",
          f"Total adaptation events: **{len(rows)}**", ""]
    emit(f"Total adaptation events: {len(rows)}")
    for etype, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        emit(f"  {etype:28} {len(items):4}")
        md.append(f"## {etype} — {len(items)} events")
        for ex in items[-2:]:
            md.append(f"- `{ex['timestamp']}` — {ex['detail']}")
        md.append("")
    path = os.path.join(EVAL_DIR, "adaptation_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    emit(f"saved {path}")


def study_pattern_growth(user_id):
    emit("")
    emit("=" * 66)
    emit("PATTERN GROWTH OVER TIME")
    emit("=" * 66)
    rows = database.get_all_conversations(user_id)
    emit(f"Conversations on record: {len(rows)}")
    if len(rows) < 3:
        emit("Too few conversations to chart growth."); return

    # The patterns table stores frequency+last_seen but not first-seen, so
    # growth is faithfully RECONSTRUCTED by replaying stored messages
    # chronologically through the same detection rules patterns.py uses.
    seen = set()
    conv_x, conv_y, pat_y = [], [], []
    for i, r in enumerate(rows, 1):
        msg = (r.get("user_message") or "").lower()
        for kw in TOPIC_KEYWORDS:
            if kw in msg:
                seen.add(("frequent_topic", kw))
        ts = (r.get("timestamp") or "")[:13]
        if len(ts) == 13:
            seen.add(("active_hour", ts[-2:]))
        if r.get("mood"):
            seen.add(("dominant_mood", r["mood"]))
        if r.get("language"):
            seen.add(("language_used", r["language"]))
        conv_x.append(i); conv_y.append(i); pat_y.append(len(seen))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(conv_x, conv_y, color=RUST, lw=2, ls=":", label="Cumulative conversations")
    ax.plot(conv_x, pat_y, color=BLUE, lw=2, label="Cumulative distinct patterns learned")
    ax.set_xlabel("Conversation #")
    ax.set_ylabel("Count")
    ax.set_title("Behavioural pattern growth with usage")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(EVAL_DIR, "pattern_growth.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    emit(f"Distinct patterns reconstructed: {len(seen)}")
    emit(f"saved {path}")


def main():
    database.init_db()
    user = database.get_first_user()
    if not user:
        emit("No user in database."); return
    emit(f"Evaluating user: {user['name']} (id={user['id']})\n")
    study_accuracy(user["id"])
    study_learning_curve(user["id"])
    study_adaptation_summary(user["id"])
    study_pattern_growth(user["id"])
    report = os.path.join(EVAL_DIR, "accuracy_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nreport saved to {report}")


if __name__ == "__main__":
    main()
