"""
okumetrik.scoring — hizalanmış okuma çıktısından akıcılık metrikleri.

Üretilen metrikler:
  total_words      : referans metindeki kelime sayısı
  words_correct    : doğru okunan kelime (telaffuz hatası YANLIŞ sayılır)
  accuracy         : words_correct / total_words
  wcpm             : dakikada doğru okunan kelime (Words Correct Per Minute)
  miscue_counts    : tür bazında sayım
  fluency_score    : 0..100 bileşik akıcılık skoru (PoC formülü; üretimde
                     uzman puanıyla kalibre edilir — bkz. İP5)

NOT: WCPM ve akıcılık eşikleri sınıf/yaş düzeyine göre değişir; burada PoC
amaçlı genel değerler kullanılmıştır. Uzman kalibrasyonu İP5 kapsamındadır.
"""

ERROR_STATUSES = {"substitution", "mispronunciation", "omission", "insertion"}


def wer(reference: str, hypothesis: str) -> dict:
    """Kelime hata oranı bileşenlerini hesapla (S/D/I/N).

    Döndürür: {"S": ikame, "D": silme, "I": ekleme, "N": ref kelime sayısı}
    """
    r = reference.split()
    h = hypothesis.split()
    N, M = len(r), len(h)
    # DP tablosu: dp[i][j] = (S, D, I) en ucuz hizalama
    INF = float("inf")
    dp = [[(INF, 0, 0, 0)] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for j in range(1, M + 1):
        dp[0][j] = (0, 0, 0, j)  # j ekleme
    for i in range(1, N + 1):
        dp[i][0] = (0, i, 0, 0)  # i silme
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if r[i - 1] == h[j - 1]:
                s, d, ins, _ = dp[i - 1][j - 1]
                dp[i][j] = (s, d, ins, s + d + ins)
            else:
                sub = dp[i - 1][j - 1]; cost_sub = sub[0] + sub[1] + sub[2] + 1
                dlt = dp[i - 1][j];     cost_del = dlt[0] + dlt[1] + dlt[2] + 1
                ins = dp[i][j - 1];     cost_ins = ins[0] + ins[1] + ins[2] + 1
                if cost_sub <= cost_del and cost_sub <= cost_ins:
                    dp[i][j] = (sub[0] + 1, sub[1], sub[2], cost_sub)
                elif cost_del <= cost_ins:
                    dp[i][j] = (dlt[0], dlt[1] + 1, dlt[2], cost_del)
                else:
                    dp[i][j] = (ins[0], ins[1], ins[2] + 1, cost_ins)
    S, D, I, _ = dp[N][M]
    return {"S": S, "D": D, "I": I, "N": N}


def summarize(entries, audio_duration_sec: float | None = None) -> dict:
    counts = {
        "correct": 0, "substitution": 0, "mispronunciation": 0,
        "omission": 0, "insertion": 0, "repetition": 0, "self_correction": 0,
    }
    for e in entries:
        if e.status in counts:
            counts[e.status] += 1

    total_words = sum(1 for e in entries if e.ref is not None and e.status != "_consumed")
    # Kendi kendine düzeltme bir hata sayılmaz (ORF konvansiyonu). Kredi:
    #  - ref'i olan self_correction (varyant A): tam kredi (sonunda doğru okundu)
    #  - ref'i olmayan self_correction (varyant B): kredi eşlik eden 'correct'ten gelir
    selfcorr_with_ref = sum(1 for e in entries if e.status == "self_correction" and e.ref is not None)
    words_correct = counts["correct"] + selfcorr_with_ref
    accuracy = words_correct / total_words if total_words else 0.0

    total_errors = (counts["substitution"] + counts["mispronunciation"]
                    + counts["omission"] + counts["insertion"])

    wcpm = None
    rate_wpm = None
    if audio_duration_sec and audio_duration_sec > 0:
        minutes = audio_duration_sec / 60.0
        wcpm = round(words_correct / minutes, 1)
        words_attempted = sum(1 for e in entries if e.hyp is not None)
        rate_wpm = round(words_attempted / minutes, 1)

    # PoC bileşik akıcılık skoru: doğruluk ağırlıklı + tekrar/duraksama cezası
    penalty = (counts["repetition"] + counts["self_correction"]) / max(total_words, 1)
    fluency_score = round(max(0.0, min(1.0, accuracy - 0.15 * penalty)) * 100, 1)

    return {
        "total_words": total_words,
        "words_correct": round(words_correct, 1),
        "accuracy": round(accuracy, 4),
        "total_errors": total_errors,
        "wcpm": wcpm,
        "rate_wpm": rate_wpm,
        "miscue_counts": counts,
        "fluency_score": fluency_score,
    }


# ---- Değerlendirme yardımcıları (altın etikete karşı F1 — Ölçüt 1/2) ----

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_miscues(pred_entries, gold_miscue_ref_indices: set) -> dict:
    """
    Ölçüt 1 (miscue F1) için basit değerlendirme:
    referans-kelime indeksi bazında, sistemin 'hata' dediği kelimeler ile
    altın (uzman) etiketteki hatalı kelimeleri karşılaştırır.
    """
    pred = {e.ref_idx for e in pred_entries
            if e.ref_idx is not None and e.status in ERROR_STATUSES}
    gold = set(gold_miscue_ref_indices)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return prf(tp, fp, fn)
