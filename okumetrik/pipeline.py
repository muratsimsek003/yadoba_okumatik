"""
okumetrik.pipeline — üst düzey okuma analizi API'si.

Gerçek sistemde 'hypothesis', HF-MTM/ASR çıktısıdır (kelime + zaman damgası).
PoC'de bu çıktı doğrudan verilebilir veya bir referanstan simüle edilebilir.
"""

from .miscue import tokenize, align
from .scoring import summarize, evaluate_miscues
from .feedback import per_word_feedback, overall_feedback


def analyze(reference_text: str, hypothesis, audio_duration_sec=None) -> dict:
    """
    reference_text : çocuğun okuması beklenen metin
    hypothesis     : ya str (boşlukla ayrılmış kelimeler) ya da
                     [{"word":..., "start":..., "end":...}, ...] listesi
    audio_duration_sec : WCPM için toplam süre (sn) — opsiyonel
    """
    ref_tokens = tokenize(reference_text)
    if isinstance(hypothesis, str):
        hyp_tokens = tokenize(hypothesis)
    else:
        hyp_tokens = [tokenize(h["word"])[0] if tokenize(h["word"]) else "" for h in hypothesis]
        hyp_tokens = [w for w in hyp_tokens if w]
        if audio_duration_sec is None and hypothesis:
            try:
                audio_duration_sec = max(h["end"] for h in hypothesis)
            except (KeyError, ValueError):
                audio_duration_sec = None

    entries = align(ref_tokens, hyp_tokens)
    metrics = summarize(entries, audio_duration_sec)
    report = {
        "metrics": metrics,
        "overall_feedback": overall_feedback(metrics),
        "word_feedback": per_word_feedback(entries),
        "tokens": [
            {"status": e.status, "ref": e.ref, "hyp": e.hyp,
             "ref_idx": e.ref_idx, "hyp_idx": e.hyp_idx,
             "detail": {k: v for k, v in e.detail.items() if k != "phoneme_ops"}}
            for e in entries
        ],
    }
    return report, entries


def evaluate(entries, gold_miscue_ref_indices) -> dict:
    """Ölçüt 1 (miscue F1) için altın etikete karşı değerlendirme."""
    return evaluate_miscues(entries, set(gold_miscue_ref_indices))
