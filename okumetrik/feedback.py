"""
okumetrik.feedback — açıklanabilir, öğretmen/veli dostu Türkçe geri bildirim.

Her miscue için "hangi kelime, neden hatalı, ne yapılmalı" biçiminde
insan-okunur açıklama üretir (İP5 — açıklanabilir geri bildirim).
"""

_PH_LABEL = {
    "a": "a", "E": "e", "I": "ı", "i": "i", "o": "o", "2": "ö", "u": "u", "y": "ü",
    "b": "b", "dZ": "c", "tS": "ç", "d": "d", "f": "f", "g": "g", "h": "h",
    "Z": "j", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "r": "r",
    "s": "s", "S": "ş", "t": "t", "v": "v", "j": "y", "z": "z", ":": "(uzatma)",
}


def _ph(p):
    return _PH_LABEL.get(p, p)


def per_word_feedback(entries) -> list[str]:
    msgs = []
    for e in entries:
        if e.status == "substitution":
            msgs.append(f'"{e.ref}" kelimesi "{e.hyp}" olarak okundu '
                        f"(farklı kelime). Kelimeyi birlikte yeniden okuyun.")
        elif e.status == "mispronunciation":
            ch = e.detail.get("changed", [])
            if ch:
                parts = []
                for a, b in ch[:3]:
                    if a and b:
                        parts.append(f"{_ph(a)}->{_ph(b)}")
                    elif a:
                        parts.append(f"{_ph(a)} sesi düştü")
                    elif b:
                        parts.append(f"{_ph(b)} sesi eklendi")
                detail = ", ".join(parts)
                msgs.append(f'"{e.ref}" telaffuz hatası ({detail}). '
                            f"Bu sesleri vurgulayarak tekrar ettirin.")
            else:
                msgs.append(f'"{e.ref}" telaffuz hatası.')
        elif e.status == "omission":
            msgs.append(f'"{e.ref}" kelimesi atlandı (okunmadı).')
        elif e.status == "insertion":
            msgs.append(f'Metinde olmayan "{e.hyp}" kelimesi eklendi.')
        elif e.status == "repetition":
            msgs.append(f'"{e.hyp}" kelimesi tekrar okundu (duraksama göstergesi).')
        elif e.status == "self_correction":
            word = e.ref or e.detail.get("corrected_to", "?")
            msgs.append(f'"{word}" önce "{e.detail.get("first_attempt")}" '
                        f"denenip kendiliğinden düzeltildi (olumlu — izleme stratejisi çalışıyor).")
    return msgs


def overall_feedback(metrics: dict) -> str:
    acc = metrics["accuracy"]
    if acc >= 0.97:
        level = "Bağımsız okuma düzeyi (çok iyi)"
    elif acc >= 0.90:
        level = "Öğretimsel düzey (desteklenmeli)"
    else:
        level = "Engellenme düzeyi (yoğun destek gerekli)"
    wcpm = metrics.get("wcpm")
    rate = f", WCPM ~ {wcpm}" if wcpm is not None else ""
    return (f"Doğruluk: %{acc*100:.1f}{rate}. Akıcılık skoru: "
            f"{metrics['fluency_score']}/100. Değerlendirme: {level}.")
