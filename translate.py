"""Step 3: MarianMT (pretrained tool) and Gemini (zero-shot, few-shot, chain-of-thought).
Usage: python translate.py [N_SENTENCES]     (default 50)   Needs env var GEMINI_API_KEY.
"""
import json, sys, time

import torch
from google import genai
from transformers import MarianMTModel, MarianTokenizer

from common import norm_hi

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
GEMINI_MODEL = "gemini-3.5-flash"          # change if your key uses another Gemini model
test = json.load(open("results/test.json", encoding="utf-8"))[:N]
srcs = [d["src"] for d in test]


def save(name, raw, sec):
    json.dump({"hyps": [norm_hi(r) for r in raw], "raw": raw, "sec": sec},
              open(f"results/{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------- MarianMT ----------
dev = "cuda" if torch.cuda.is_available() else "cpu"
name = "Helsinki-NLP/opus-mt-en-hi"
tk, mm = MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).to(dev).eval()
raw, t0 = [], time.time()
with torch.no_grad():
    for i in range(0, N, 16):
        enc = tk(srcs[i:i + 16], return_tensors="pt", padding=True).to(dev)
        raw += tk.batch_decode(mm.generate(**enc, num_beams=4, max_new_tokens=100), skip_special_tokens=True)
save("marian", raw, (time.time() - t0) / N)
print("MarianMT done")

# ---------- Gemini ----------
client = genai.Client()                     # reads GEMINI_API_KEY
shots = json.load(open("results/fewshot.json", encoding="utf-8"))
examples = "\n".join(f"English: {s['en']}\nHindi: {s['hi']}\n" for s in shots)
PROMPTS = {
    "gemini_zero": "Translate this English sentence into Hindi (Devanagari). Output only the translation.\n\nEnglish: {s}",
    "gemini_few": "Translate English into Hindi (Devanagari). Output only the translation.\n\n" + examples + "\nEnglish: {s}\nHindi:",
    "gemini_cot": "Translate this English sentence into Hindi (Devanagari). First think step by step about key words, "
                  "word order and tense. Then end with a last line 'Translation: <Hindi sentence>'.\n\nEnglish: {s}",
}


def ask(prompt):
    for i in range(5):
        try:
            text = client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()
            time.sleep(4)                   # stay under free-tier rate limits
            return text
        except Exception as e:
            print("retry:", str(e)[:100]); time.sleep(10 * (i + 1))
    return ""


for key, template in PROMPTS.items():
    raw, t0 = [], time.time()
    for i, s in enumerate(srcs):
        out = ask(template.format(s=s))
        if key == "gemini_cot":
            out = out.split("Translation:")[-1].strip()
        raw.append(out.splitlines()[0] if out else "")
        if (i + 1) % 10 == 0:
            print(f"{key}: {i + 1}/{N}")
    save(key, raw, (time.time() - t0 - 4 * N) / N)    # subtract the rate-limit sleeps
print("Next: python compare.py")
