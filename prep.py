"""Step 1: load data, clean, split 80/10/10, build vocab, POS tag + dependency parse."""
import json, os, pickle, random
from collections import Counter

import matplotlib.pyplot as plt
import spacy
from datasets import load_dataset

from common import clean_en, clean_hi, tok

N_RAW, MAX_LEN, MIN_FREQ = 30000, 20, 2
random.seed(42)
os.makedirs("results", exist_ok=True)

# ---- load, clean, filter ----
ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", split=f"train[:{N_RAW}]")
data, seen = [], set()
for d in ds:
    raw, hi_raw = d["translation"]["en"], d["translation"]["hi"]
    en, hi = tok(clean_en(raw)), tok(clean_hi(hi_raw))
    if 3 <= len(en) <= MAX_LEN and 3 <= len(hi) <= MAX_LEN and raw not in seen:
        seen.add(raw)
        data.append({"raw": raw, "hi_raw": hi_raw, "en": en, "hi": hi})
random.shuffle(data)
n = len(data)
train, val, test = data[:int(.8 * n)], data[int(.8 * n):int(.9 * n)], data[int(.9 * n):]
print(f"{n} pairs -> train {len(train)} | val {len(val)} | test {len(test)}")


# ---- vocab (train only) ----
def vocab(sents):
    c = Counter(w for s in sents for w in s)
    return ["<pad>", "<sos>", "<eos>", "<unk>"] + [w for w, f in c.items() if f >= MIN_FREQ]


src_itos, tgt_itos = vocab([d["en"] for d in train]), vocab([d["hi"] for d in train])
pickle.dump({"train": train, "val": val, "test": test, "src_itos": src_itos, "tgt_itos": tgt_itos},
            open("results/data.pkl", "wb"))
json.dump([{"src": d["raw"], "ref": " ".join(d["hi"])} for d in test],
          open("results/test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump([{"en": d["raw"], "hi": d["hi_raw"]} for d in train if 5 <= len(d["en"]) <= 10][:3],
          open("results/fewshot.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"vocab: English {len(src_itos)} | Hindi {len(tgt_itos)}")

with open("results/stats.tex", "w") as f:  # dataset table for the report
    f.write("\\begin{tabular}{lr}\n\\toprule\nItem & Count \\\\\n\\midrule\n"
            f"Sentence pairs after cleaning & {n} \\\\\nTrain / Val / Test & {len(train)} / {len(val)} / {len(test)} \\\\\n"
            f"English vocabulary & {len(src_itos)} \\\\\nHindi vocabulary & {len(tgt_itos)} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ---- POS tagging + dependency parse (spaCy) ----
nlp = spacy.load("en_core_web_sm")
pos = Counter(t.pos_ for doc in nlp.pipe([d["raw"] for d in train[:1000]]) for t in doc if not t.is_space)
plt.figure(figsize=(7, 3.5))
plt.bar(*zip(*pos.most_common()))
plt.xticks(rotation=45); plt.title("POS tag distribution (1000 training sentences)")
plt.tight_layout(); plt.savefig("results/pos_distribution.png", dpi=200)
print("POS counts:", dict(pos))

sample = nlp(" ".join(next(d["en"] for d in train if 7 <= len(d["en"]) <= 10)))
rows = "\n".join(f"{t.text} & {t.pos_} & {t.dep_} & {t.head.text} \\\\" for t in sample)
open("results/dep_table.tex", "w").write(
    "\\begin{tabular}{llll}\n\\toprule\nWord & POS & Dependency & Head \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n")
print("Parsed sentence:", sample.text)
print("Next: python train.py")
