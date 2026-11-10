"""Shared helpers: cleaning, tokenizing, scoring (English -> Hindi)."""
import re
from collections import Counter

import sacrebleu

CONTRACTIONS = [("won't", "will not"), ("can't", "cannot"), ("n't", " not"), ("'re", " are"),
                ("'ve", " have"), ("'ll", " will"), ("'m", " am"), ("it's", "it is")]


def clean_en(s):
    s = s.lower()
    for a, b in CONTRACTIONS:
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9\s.,?!]", " ", s)


def clean_hi(s):
    return re.sub(r"[^\u0900-\u097F0-9\s.,?!]", " ", s)  # keep Devanagari, digits, punctuation


def tok(s):
    # split punctuation, then whitespace (\w would break Hindi words at vowel signs)
    return re.sub(r"([.,?!।])", r" \1 ", s).split()


def norm_hi(s):
    """Same normalisation for references and every system output before scoring."""
    return " ".join(tok(clean_hi(s)))


def rouge1(h, r):
    h, r = Counter(h.split()), Counter(r.split())
    ov = sum((h & r).values())
    if ov == 0:
        return 0.0
    p, rc = ov / sum(h.values()), ov / sum(r.values())
    return 2 * p * rc / (p + rc)


def score(hyps, refs):
    return {"BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
            "ROUGE-1": round(100 * sum(rouge1(h, r) for h, r in zip(hyps, refs)) / len(refs), 2)}
