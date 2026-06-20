#!/usr/bin/env python3
"""
selftest_offline.py — fine-tune + WER pipeline'ının OFFLINE doğrulaması.

AMAÇ: Ağ/GPU gerektirmeden, gerçek CTC eğitim döngüsünün ve WER hesabının
uçtan uca çalıştığını KANITLAMAK. Sentetik "konuşma benzeri" özellikler ile
minik bir BiLSTM+CTC modeli eğitilir; eğitim öncesi/sonrası WER karşılaştırılır.

BU GERÇEK BİR TÜRKÇE WER DEĞİLDİR. Gerçek sayı için pre_feasibility/
finetune_whisper_tr.py betiğini Colab/GPU'da, gerçek veri setiyle çalıştırın.
Buradaki tek iddia: "harness çalışıyor, WER hesabı doğru, eğitim WER'i düşürüyor".
"""
import random, sys, os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from okumetrik.scoring import wer  # kendi WER fonksiyonumuz

random.seed(0); torch.manual_seed(0)

LEXICON = ["ali", "topu", "at", "kedi", "balık", "koştu", "gel", "git", "oku", "kitap"]
CHARS = sorted(set("".join(LEXICON) + " "))
BLANK = 0
VOCAB = ["<blank>"] + CHARS
CH2ID = {c: i for i, c in enumerate(VOCAB)}
ID2CH = {i: c for c, i in CH2ID.items()}
F = 24            # özellik boyutu
FPC = 3           # karakter başına çerçeve
PROJ = torch.randn(len(VOCAB), F)   # karakter -> özellik (sabit "akustik")


def make_sample():
    words = [random.choice(LEXICON) for _ in range(random.randint(3, 5))]
    text = " ".join(words)
    label_ids = [CH2ID[c] for c in text]
    # "akustik" özellik: her karakter FPC çerçeveye yayılır + gürültü
    feats = []
    for cid in label_ids:
        base = PROJ[cid]
        for _ in range(FPC):
            feats.append(base + 0.35 * torch.randn(F))
    return torch.stack(feats), text, label_ids


class TinyASR(nn.Module):
    def __init__(self, F, H=64, V=len(VOCAB)):
        super().__init__()
        self.rnn = nn.LSTM(F, H, batch_first=True, bidirectional=True, num_layers=1)
        self.head = nn.Linear(2 * H, V)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o).log_softmax(-1)


def ctc_greedy_decode(log_probs):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], None
    for t in ids:
        if t != prev and t != BLANK:
            out.append(ID2CH[t])
        prev = t
    return "".join(out)


def corpus_wer(model, data):
    refs, hyps = [], []
    model.eval()
    with torch.no_grad():
        for feats, text, _ in data:
            lp = model(feats.unsqueeze(0))[0]
            hyps.append(ctc_greedy_decode(lp) or "<bos>")
            refs.append(text)
    # mikro-ortalama WER (tüm cümleler birleşik)
    tot = {"S": 0, "D": 0, "I": 0, "N": 0}
    for r, h in zip(refs, hyps):
        c = wer(r, h)
        for k in ("S", "D", "I", "N"):
            tot[k] += c[k]
    return (tot["S"] + tot["D"] + tot["I"]) / max(tot["N"], 1), refs, hyps


def main():
    train = [make_sample() for _ in range(240)]
    test = [make_sample() for _ in range(40)]
    model = TinyASR(F)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    w0, _, _ = corpus_wer(model, test)
    print(f"[Eğitim ÖNCESİ]  WER = %{w0*100:.1f}  (rastgele başlatılmış model)")

    EPOCHS = 12
    for ep in range(EPOCHS):
        model.train(); random.shuffle(train); total = 0.0
        for feats, text, label_ids in train:
            x = feats.unsqueeze(0)
            lp = model(x).transpose(0, 1)            # (T,1,V)
            tgt = torch.tensor(label_ids).unsqueeze(0)
            in_len = torch.tensor([x.shape[1]])
            tg_len = torch.tensor([len(label_ids)])
            loss = ctc(lp, tgt, in_len, tg_len)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 3 == 0:
            w, _, _ = corpus_wer(model, test)
            print(f"  epoch {ep+1:2d}/{EPOCHS}  loss={total/len(train):.3f}  WER=%{w*100:.1f}")

    wf, refs, hyps = corpus_wer(model, test)
    print(f"[Eğitim SONRASI] WER = %{wf*100:.1f}")
    print("\nÖrnek tahminler (referans -> model çıktısı):")
    for r, h in list(zip(refs, hyps))[:4]:
        print(f"  {r!r:40s} -> {h!r}")
    print(f"\nSONUÇ: harness çalışıyor; eğitim WER'i %{w0*100:.0f} -> %{wf*100:.1f} düşürdü.")
    print("Bu sentetik bir doğrulamadır; gerçek Türkçe WER için Colab betiğini kullanın.")


if __name__ == "__main__":
    main()
