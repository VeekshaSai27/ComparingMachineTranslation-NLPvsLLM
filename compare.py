"""Step 4: score every system on the same sentences, write tables and a chart for the report."""
import json

import matplotlib.pyplot as plt

from common import score

R = "results/"
FILES = {"Seq2Seq GRU (from scratch)": "baseline", "MarianMT (pretrained tool)": "marian",
         "Gemini zero-shot": "gemini_zero", "Gemini few-shot": "gemini_few", "Gemini chain-of-thought": "gemini_cot"}
S = {}
for name, f in FILES.items():
    try:
        S[name] = json.load(open(R + f + ".json", encoding="utf-8"))
    except FileNotFoundError:
        print("missing", f)
test = json.load(open(R + "test.json", encoding="utf-8"))
N = min(len(s["hyps"]) for s in S.values())
refs = [d["ref"] for d in test[:N]]
print(f"Comparing on {N} test sentences\n")

rows = []
for name, s in S.items():
    sc = score(s["hyps"][:N], refs)
    ratio = sum(len(h.split()) for h in s["hyps"][:N]) / sum(len(r.split()) for r in refs)
    rows.append((name, sc["BLEU"], sc["chrF"], sc["ROUGE-1"], round(ratio, 2), round(1000 * s["sec"], 1)))
    print(rows[-1])

with open(R + "results_table.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nSystem & BLEU & chrF & ROUGE-1 & Length ratio & ms/sentence \\\\\n\\midrule\n")
    for r in rows:
        f.write(" & ".join(map(str, r)) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

plt.figure(figsize=(7, 3.5))
x = range(len(rows))
plt.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, label="BLEU")
plt.bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4, label="chrF")
plt.xticks(list(x), [r[0].replace(" (", "\n(") for r in rows], fontsize=6); plt.legend(); plt.tight_layout()
plt.savefig(R + "comparison_bar.png", dpi=200)


def esc(s):
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_")]:
        s = s.replace(a, b)
    return s


with open(R + "samples_table.tex", "w", encoding="utf-8") as f:  # first 5 sentences, side by side
    f.write("\\begin{longtable}{p{0.22\\linewidth}p{0.72\\linewidth}}\n\\toprule\n")
    for i in range(min(5, N)):
        f.write(f"\\textbf{{Source {i + 1}}} & {esc(test[i]['src'])} \\\\\nReference & {{\\hindifont {esc(refs[i])}}} \\\\\n")
        for name, s in S.items():
            f.write(f"{esc(name.split(' (')[0])} & {{\\hindifont {esc(s['hyps'][i])}}} \\\\\n")
        f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{longtable}\n")
print("\nDone: results/*.tex and charts written. Compile report.tex with XeLaTeX.")
