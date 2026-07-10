"""Bundle the evaluation outputs into one openable HTML page: report.html"""
import os, html

HERE = os.path.dirname(os.path.abspath(__file__))

def read(name):
    p = os.path.join(HERE, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else "(not generated yet)"

acc = html.escape(read("accuracy_report.txt"))
adapt = html.escape(read("adaptation_summary.md"))

page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ARIA — Evaluation Report</title>
<style>
 body {{ font-family: Georgia, serif; background:#f3ebdc; color:#3c2a1e; max-width:900px; margin:2rem auto; padding:0 1rem; }}
 h1,h2 {{ color:#7a4a2e; }}
 pre {{ background:#fbf7ee; border:1px solid #d8c9b0; border-radius:8px; padding:1rem; overflow-x:auto; font-size:13px; }}
 img {{ max-width:100%; border:1px solid #d8c9b0; border-radius:8px; margin:.5rem 0; }}
</style></head><body>
<h1>ARIA — Behavioural Learning Evaluation</h1>
<p>Generated from the real database. Re-generate anytime with:
<code>.venv\\Scripts\\python.exe evaluate_accuracy.py</code></p>
<h2>1. Accuracy report (fused &amp; per-modality vs ground truth)</h2>
<pre>{acc}</pre>
<h2>2. Learning curve — personal KNN vs static rules</h2>
<img src="learning_curve.png" alt="learning curve">
<h2>3. Pattern growth with usage</h2>
<img src="pattern_growth.png" alt="pattern growth">
<h2>4. Confusion matrix</h2>
<p><img src="confusion_matrix.png" alt="confusion matrix (appears once ground-truth labels exist)"
     onerror="this.replaceWith('Not generated yet — needs ground-truth labels (say: label happy / label stressed to ARIA).')"></p>
<h2>5. Adaptation ledger summary</h2>
<pre>{adapt}</pre>
</body></html>"""

out = os.path.join(HERE, "report.html")
open(out, "w", encoding="utf-8").write(page)
print("wrote", out)
