"""Step 2: train a small GRU encoder-decoder with dot-product attention, then test it."""
import json, math, pickle, random, time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from common import norm_hi, score

H, BATCH, EPOCHS, LR = 256, 128, 10, 1e-3
PAD, SOS, EOS, UNK = 0, 1, 2, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42); torch.manual_seed(42)

D = pickle.load(open("results/data.pkl", "rb"))
src_itos, tgt_itos = D["src_itos"], D["tgt_itos"]
s2i = {w: i for i, w in enumerate(src_itos)}
t2i = {w: i for i, w in enumerate(tgt_itos)}


def collate(batch):
    pad = lambda xs: pad_sequence([torch.tensor(x) for x in xs], batch_first=True)
    return pad([b[0] for b in batch]), pad([b[1] for b in batch])


def loader(split, shuffle):
    pairs = [([s2i.get(w, UNK) for w in d["en"]] + [EOS],
              [SOS] + [t2i.get(w, UNK) for w in d["hi"]] + [EOS]) for d in D[split]]
    return DataLoader(pairs, BATCH, shuffle=shuffle, collate_fn=collate)


train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)


class Seq2Seq(nn.Module):
    def __init__(s):
        super().__init__()
        s.se, s.te = nn.Embedding(len(src_itos), H, padding_idx=PAD), nn.Embedding(len(tgt_itos), H, padding_idx=PAD)
        s.enc, s.dec = nn.GRU(H, H, batch_first=True), nn.GRU(H, H, batch_first=True)
        s.out, s.drop = nn.Linear(2 * H, len(tgt_itos)), nn.Dropout(0.3)

    def encode(s, src):
        return s.enc(s.drop(s.se(src)))                      # enc: B,L,H  h: 1,B,H

    def step(s, tok, h, enc, mask):
        o, h = s.dec(s.drop(s.te(tok)).unsqueeze(1), h)       # o: B,1,H
        a = torch.bmm(o, enc.transpose(1, 2)).masked_fill(~mask.unsqueeze(1), -1e9).softmax(2)  # B,1,L
        ctx = torch.bmm(a, enc)                               # B,1,H
        return s.out(torch.cat([o, ctx], 2)).squeeze(1), h    # B,V

    def forward(s, src, tgt, tf):
        enc, h = s.encode(src)
        mask, tok, outs = src != PAD, tgt[:, 0], []
        for t in range(1, tgt.size(1)):
            lg, h = s.step(tok, h, enc, mask)
            outs.append(lg)
            tok = tgt[:, t] if random.random() < tf else lg.argmax(1)
        return torch.stack(outs, 1)                           # B,T-1,V


model = Seq2Seq().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss(ignore_index=PAD)
n_params = sum(p.numel() for p in model.parameters())
print(f"device={dev} params={n_params / 1e6:.1f}M")


def run(dl, train):
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for src, tgt in dl:
            src, tgt = src.to(dev), tgt.to(dev)
            out = model(src, tgt, 0.5 if train else 1.0)
            loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
    return total / len(dl)


tr_hist, va_hist, best, t0 = [], [], 1e9, time.time()
for ep in range(1, EPOCHS + 1):
    tl, vl = run(train_dl, True), run(val_dl, False)
    tr_hist.append(tl); va_hist.append(vl)
    if vl < best:
        best = vl; torch.save(model.state_dict(), "results/seq2seq.pt")
    print(f"epoch {ep:2d} | train loss {tl:.3f} | val loss {vl:.3f} | val perplexity {math.exp(vl):.1f}")
train_time = time.time() - t0

plt.figure(figsize=(5, 3.5))
plt.plot(tr_hist, label="train"); plt.plot(va_hist, label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=200)

# ---- test: greedy decoding ----
model.load_state_dict(torch.load("results/seq2seq.pt", map_location=dev))
model.eval()
hyps, t0 = [], time.time()
with torch.no_grad():
    for src, _ in test_dl:
        src = src.to(dev)
        enc, h = model.encode(src)
        mask, tok, outs = src != PAD, torch.full((src.size(0),), SOS, device=dev), []
        for _ in range(30):
            lg, h = model.step(tok, h, enc, mask)
            tok = lg.argmax(1); outs.append(tok)
        for row in torch.stack(outs, 1).tolist():
            row = row[:row.index(EOS)] if EOS in row else row
            hyps.append(norm_hi(" ".join(tgt_itos[i] for i in row if i != UNK)))
sec = (time.time() - t0) / len(hyps)

refs = [r["ref"] for r in json.load(open("results/test.json", encoding="utf-8"))]
sc = score(hyps, refs)
json.dump({"hyps": hyps, "sec": sec, "params_M": n_params / 1e6}, open("results/baseline.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
with open("results/baseline_table.tex", "w") as f:
    f.write("\\begin{tabular}{lr}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
            f"Best validation loss & {best:.3f} \\\\\nValidation perplexity & {math.exp(best):.1f} \\\\\n"
            f"Test BLEU & {sc['BLEU']} \\\\\nTest chrF & {sc['chrF']} \\\\\n"
            f"Training time (s) & {train_time:.0f} \\\\\nInference (ms/sentence) & {1000 * sec:.1f} \\\\\n"
            f"Parameters (M) & {n_params / 1e6:.1f} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
print("test scores:", sc)
print("Next: python translate.py 50")
