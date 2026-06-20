"""
asr_scaffold.model — HF-MTM akustik omurgası (wav2vec2 + CTC fonem tanıma).

Bu modül, projenin özgün HF-MTM yönteminin AKUSTIK katmanıdır:
çocuk konuşmasından fonem-düzeyi olasılıklar (posteriors) üretir. Çıktı,
hizalama-serbest çözümleme (decode_free) ile bir fonem dizisine dönüştürülür;
bu dizi okumetrik paketindeki sözcük/fonem hizalama ve miscue mantığını besler.

GEREKSİNİM: GPU + PyTorch + transformers + (opsiyonel) peft (LoRA).
Bu dosya bu ortamda EĞİTİLMEZ; eğitim için gerçek çocuk-okuma korpusu (İP3)
ve GPU gerekir. Kod, böyle bir ortamda çalışacak şekilde yapılandırılmıştır.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2Config
except Exception as _e:  # ortamda torch/transformers yoksa içe aktarma sorunsuz geçer
    torch = None
    nn = object  # type: ignore


# Basit Türkçe fonem alfabesi (okumetrik.g2p ile uyumlu) + CTC boşluğu
PHONEME_VOCAB = ["<blank>", "a","E","I","i","o","2","u","y",
                 "b","dZ","tS","d","f","g","h","Z","k","l","m","n",
                 "p","r","s","S","t","v","j","z",":"]
PH2ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
ID2PH = {i: p for p, i in PH2ID.items()}


class HFMTMAcoustic(nn.Module):
    """wav2vec2 omurga + doğrusal CTC fonem başı."""

    def __init__(self, pretrained="facebook/wav2vec2-xls-r-300m",
                 vocab_size=len(PHONEME_VOCAB), freeze_feature_encoder=True,
                 use_lora=False):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained)
        if freeze_feature_encoder:
            self.backbone.feature_extractor._freeze_parameters()
        if use_lora:
            self._inject_lora()
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.ctc_head = nn.Linear(hidden, vocab_size)

    def _inject_lora(self):
        """LoRA adaptörleri (düşük kaynaklı verimli ince ayar — Koçak & Ulaş, 2024)."""
        try:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "v_proj"])
            self.backbone = get_peft_model(self.backbone, cfg)
        except Exception as e:
            print(f"[uyarı] LoRA atlandı: {e}")

    def forward(self, input_values, attention_mask=None):
        out = self.backbone(input_values, attention_mask=attention_mask).last_hidden_state
        logits = self.ctc_head(self.dropout(out))          # (B, T, V)
        return logits.log_softmax(dim=-1)

    @torch.no_grad() if torch else (lambda f: f)
    def decode_free(self, log_probs):
        """
        Hizalama-serbest CTC greedy çözümleme: fonem-sınırı varsaymadan
        bir fonem dizisi üretir (HF-MTM'in 'alignment-free' yönü).
        """
        ids = log_probs.argmax(dim=-1).tolist()
        out, prev = [], None
        for t in ids:
            if t != prev and t != PH2ID["<blank>"]:
                out.append(ID2PH[t])
            prev = t
        return out
