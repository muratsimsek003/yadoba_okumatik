"""
asr_scaffold.train — HF-MTM akustik modelinin CTC ile eğitimi.

Özellikler:
  - CTC kaybı ile fonem tanıma eğitimi
  - Çocuk-konuşması veri artırımı (augment.py) açma/kapama
  - LoRA ile düşük kaynaklı verimli ince ayar (--lora)
  - Aktif öğrenme seçim kancası (entropy tabanlı) — az etiketli senaryo (İP3/İP4)

ÇALIŞTIRMA (GPU'lu makinede):
  pip install -r requirements.txt
  python train.py --manifest data/train.jsonl --val data/val.jsonl --lora --augment

Bu ortamda ÇALIŞTIRILMAZ (torch/transformers/GPU ve İP3 verisi gerekir).
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def entropy_select(model, pool_loader, k, device):
    """Aktif öğrenme: en yüksek belirsizlikli (entropi) k örneği seçer."""
    import torch
    scores = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(pool_loader):
            lp = model(batch["input_values"].to(device))      # (B,T,V) log-prob
            p = lp.exp()
            ent = -(p * lp).sum(-1).mean(-1)                   # ortalama çerçeve entropisi
            for b in range(ent.size(0)):
                scores.append((ent[b].item(), i * pool_loader.batch_size + b))
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:k]]


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from transformers import Wav2Vec2Processor
    from model import HFMTMAcoustic, PH2ID
    from data import ChildReadingDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained_processor)
    train_ds = ChildReadingDataset(args.manifest, args.root, processor,
                                   augment=args.augment)

    def collate(batch):
        xs = pad_sequence([b["input_values"] for b in batch], batch_first=True)
        ys = pad_sequence([b["labels"] for b in batch], batch_first=True,
                          padding_value=PH2ID["<blank>"])
        in_len = torch.tensor([b["input_values"].shape[0] for b in batch])
        tgt_len = torch.tensor([len(b["labels"]) for b in batch])
        return xs, ys, in_len, tgt_len

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate)
    model = HFMTMAcoustic(use_lora=args.lora).to(device)
    ctc = torch.nn.CTCLoss(blank=PH2ID["<blank>"], zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for xs, ys, in_len, tgt_len in loader:
            xs, ys = xs.to(device), ys.to(device)
            log_probs = model(xs).transpose(0, 1)             # (T,B,V) CTC ister
            feat_len = (in_len // 320).clamp(min=1)            # ~stride
            loss = ctc(log_probs, ys, feat_len.to(device), tgt_len.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {epoch+1}/{args.epochs}  loss={total/len(loader):.4f}")
        torch.save(model.state_dict(), os.path.join(args.out, f"hfmtm_ep{epoch+1}.pt"))


def build_argparser():
    ap = argparse.ArgumentParser(description="HF-MTM CTC eğitimi")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--pretrained_processor", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--augment", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
