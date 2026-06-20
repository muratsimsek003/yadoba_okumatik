"""
asr_scaffold.data — çocuk sesli-okuma veri seti yükleyici (manifest tabanlı).

Manifest formatı (JSONL), her satır:
  {"audio": "wav/0001.wav", "text": "küçük kırmızı balık", "phonemes": "k y tS y k ..."}

'phonemes' yoksa okumetrik.g2p ile otomatik üretilir (zayıf etiket / az-etiketli
senaryo). Bu, İP3 çıktısı veri seti üzerinde çalışır.

GEREKSİNİM: torch + torchaudio + transformers (Wav2Vec2Processor). GPU önerilir.
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from okumetrik.g2p import g2p
from .model import PH2ID
from .augment import child_like_augment

try:
    import torch, torchaudio
    import numpy as np
    from torch.utils.data import Dataset
except Exception:
    torch = None
    Dataset = object  # type: ignore


def text_to_phoneme_ids(text: str) -> list[int]:
    ids = []
    for word in text.split():
        for ph in g2p(word):
            if ph in PH2ID:
                ids.append(PH2ID[ph])
    return ids


class ChildReadingDataset(Dataset):
    def __init__(self, manifest_path, root="", processor=None,
                 augment=False, target_sr=16000):
        self.items = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
        self.root = root
        self.processor = processor
        self.augment = augment
        self.target_sr = target_sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        wav, sr = torchaudio.load(os.path.join(self.root, it["audio"]))
        wav = wav.mean(0)  # mono
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if self.augment:
            wav = torch.from_numpy(
                child_like_augment(wav.numpy(), self.target_sr)).float()
        if "phonemes" in it and it["phonemes"]:
            labels = [PH2ID[p] for p in it["phonemes"].split() if p in PH2ID]
        else:
            labels = text_to_phoneme_ids(it["text"])  # zayıf etiket
        feats = self.processor(wav.numpy(), sampling_rate=self.target_sr,
                               return_tensors="pt").input_values[0] \
            if self.processor else wav
        return {"input_values": feats, "labels": torch.tensor(labels),
                "text": it["text"]}
