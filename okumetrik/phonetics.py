"""
okumetrik.phonetics — fonem dizileri arasında mesafe ve hizalama.

İki kelimenin fonem dizileri arasındaki düzenleme mesafesini hesaplar ve
hangi fonemlerin değiştiğini döndürür. Bu, bir okuma hatasının "tamamen farklı
bir kelime (substitution)" mı yoksa "telaffuz hatası (mispronunciation)" mı
olduğunu ayırt etmek için kullanılır.

İKİ MESAFE MODELİ
-----------------
"class"   : ünlü/ünsüz sınıfı temelli kaba mesafe (ilk sürüm, baz çizgi)
"feature" : ayırt edici fonetik özellik (çıkış yeri / çıkış biçimi / ötümlülük)
            temelli mesafe — VARSAYILAN

Model seçimi ablasyon deneyiyle yapılmıştır (bkz. ablation/README.md).
İlkokul düzeyi 65 cümleden türetilen 257 sözcüklük dağarcık üzerinde, sözcük
değiştirmelerin hedefe fonetik olarak en yakın gerçek sözcüklerden seçildiği
zorlayıcı koşulda, üç ana sınıfta Macro-F1:

    klasik sabit maliyetli hizalama .................. 0,4631
    yazımsal (karakter Levenshtein) mesafe ........... 0,9317
    fonem SINIFI temelli mesafe  (eşik 0,34) ......... 0,9067
    ayırt edici ÖZELLİK temelli mesafe (eşik 0,20) ... 0,9422

Bulgu: Türkçenin sığ ortografisi nedeniyle harf mesafesi zaten fonem mesafesine
yaklaşmaktadır; kaba ünlü/ünsüz sınıflandırması bilgi kaybettirdiği için yazımsal
baz çizgiyi geçememektedir. Üstünlük ancak fonemin iç özellik yapısı (b/p yalnız
ötümlülükte, b/k yer + ötümlülükte farklıdır) kullanıldığında elde edilmektedir.

NOT: Aşağıdaki ağırlıklar ve eşik, SENTETİK hata korpusunda elde edilen ÖN
değerlerdir. Projede (İP3) uzman etiketli çocuk okuma verisinden, katılımcı-
dışarıda-bırakmalı çapraz doğrulamayla öğrenilecektir.
"""

from .g2p import g2p, is_vowel_phoneme

# ---------------------------------------------------------------------------
# Mesafe modeli seçimi
# ---------------------------------------------------------------------------
DISTANCE_MODE = "feature"          # "feature" | "class"

# Telaffuz sapması / sözcük değiştirme karar eşiği (mesafe modeline bağlı).
MISPRON_THRESHOLD = {"feature": 0.20, "class": 0.34}

# ---------------------------------------------------------------------------
# Türkçe fonem özellik tabloları
# ---------------------------------------------------------------------------
# Ünlüler: (önlük, yuvarlaklık, darlık) — üçü de ikili
VOWEL_FEATURES = {
    "a": (0, 0, 1), "E": (1, 0, 1), "I": (0, 0, 0), "i": (1, 0, 0),
    "o": (0, 1, 1), "2": (1, 1, 1), "u": (0, 1, 0), "y": (1, 1, 0),
}

# Ünsüzler: (çıkış yeri 0-6, çıkış biçimi 0-4, ötümlülük 0/1)
#   çıkış yeri  : 0 dudak, 1 diş-dudak, 2 diş eti, 3 damak, 4 art damak,
#                 5 gırtlak, 6 (yumuşak g / uzatma)
#   çıkış biçimi: 0 patlamalı, 1 sızmalı, 2 patlamalı-sızmalı, 3 genizsil, 4 akıcı
CONSONANT_FEATURES = {
    "p": (0, 0, 0), "b": (0, 0, 1), "m": (0, 3, 1),
    "f": (1, 1, 0), "v": (1, 1, 1),
    "t": (2, 0, 0), "d": (2, 0, 1), "n": (2, 3, 1),
    "s": (2, 1, 0), "z": (2, 1, 1), "l": (2, 4, 1), "r": (2, 4, 1),
    "S": (3, 1, 0), "Z": (3, 1, 1), "tS": (3, 2, 0), "dZ": (3, 2, 1),
    "j": (3, 4, 1),
    "k": (4, 0, 0), "g": (4, 0, 1),
    "h": (5, 1, 0),
    ":": (6, 4, 1),
}

N_PLACE = 6  # çıkış yeri ekseninin normalizasyon böleni

# Öğrenilebilir ağırlıklar. İP3'te uzman etiketli kalibrasyon verisinden
# yeniden kestirilecektir; serbest parametre sayısı bilinçli olarak azdır.
WEIGHTS = {
    "cross_class":  1.00,   # ünlü <-> ünsüz değişimi
    "vowel_base":   0.20,   # ünlüler arası taban maliyet
    "vowel_span":   0.64,   # ünlü özellik farkının katkısı
    "cons_base":    0.15,   # ünsüzler arası taban maliyet
    "cons_place":   0.45,   # çıkış yeri farkının katkısı
    "cons_manner":  0.30,   # çıkış biçimi farkının katkısı
    "cons_voice":   0.20,   # ötümlülük farkının katkısı
    "unknown":      0.80,   # tabloda bulunmayan fonem çifti
    "gap":          1.00,   # ekleme / düşme maliyeti
    "class_same":   0.60,   # ("class" modeli) aynı sınıf içi değişim
}


# ---------------------------------------------------------------------------
# Yerine koyma maliyetleri
# ---------------------------------------------------------------------------
def _sub_cost_class(a: str, b: str) -> float:
    """Kaba sınıf temelli maliyet (ilk sürüm; baz çizgi olarak korunur)."""
    if a == b:
        return 0.0
    if is_vowel_phoneme(a) != is_vowel_phoneme(b):
        return WEIGHTS["cross_class"]
    return WEIGHTS["class_same"]


def _sub_cost_feature(a: str, b: str) -> float:
    """Ayırt edici fonetik özellik temelli maliyet."""
    if a == b:
        return 0.0
    av, bv = a in VOWEL_FEATURES, b in VOWEL_FEATURES
    if av != bv:
        return WEIGHTS["cross_class"]
    if av:
        fa, fb = VOWEL_FEATURES[a], VOWEL_FEATURES[b]
        ndiff = sum(x != y for x, y in zip(fa, fb))
        return WEIGHTS["vowel_base"] + WEIGHTS["vowel_span"] * (ndiff / 3.0)
    fa, fb = CONSONANT_FEATURES.get(a), CONSONANT_FEATURES.get(b)
    if fa is None or fb is None:
        return WEIGHTS["unknown"]
    d_place = abs(fa[0] - fb[0]) / N_PLACE
    d_manner = 0.0 if fa[1] == fb[1] else 1.0
    d_voice = 0.0 if fa[2] == fb[2] else 1.0
    return min(1.0, WEIGHTS["cons_base"]
               + WEIGHTS["cons_place"] * d_place
               + WEIGHTS["cons_manner"] * d_manner
               + WEIGHTS["cons_voice"] * d_voice)


def _sub_cost(a: str, b: str, mode: str | None = None) -> float:
    mode = mode or DISTANCE_MODE
    return _sub_cost_feature(a, b) if mode == "feature" else _sub_cost_class(a, b)


def mispron_threshold(mode: str | None = None) -> float:
    """Aktif mesafe modeline karşılık gelen telaffuz sapması eşiği."""
    return MISPRON_THRESHOLD[mode or DISTANCE_MODE]


# ---------------------------------------------------------------------------
# Hizalama
# ---------------------------------------------------------------------------
def phoneme_alignment(pa: list[str], pb: list[str], mode: str | None = None):
    """
    Needleman-Wunsch ile iki fonem dizisini hizalar.
    Döndürür: (mesafe, ops) — ops: [('=',a,b)|('~',a,b)|('-',a,None)|('+',None,b)]
    """
    n, m = len(pa), len(pb)
    GAP = WEIGHTS["gap"]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP
        bt[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * GAP
        bt[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = _sub_cost(pa[i - 1], pb[j - 1], mode)
            diag = dp[i - 1][j - 1] + c
            up = dp[i - 1][j] + GAP
            left = dp[i][j - 1] + GAP
            best = min(diag, up, left)
            dp[i][j] = best
            bt[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "diag":
            a, b = pa[i - 1], pb[j - 1]
            ops.append(("=" if a == b else "~", a, b))
            i, j = i - 1, j - 1
        elif move == "up":
            ops.append(("-", pa[i - 1], None))
            i -= 1
        else:
            ops.append(("+", None, pb[j - 1]))
            j -= 1
    ops.reverse()
    return dp[n][m], ops


def phonetic_distance(word_a: str, word_b: str, mode: str | None = None) -> float:
    """İki kelime arasındaki normalize fonetik mesafe (0..1)."""
    pa, pb = g2p(word_a), g2p(word_b)
    if not pa and not pb:
        return 0.0
    dist, _ = phoneme_alignment(pa, pb, mode)
    return dist / max(len(pa), len(pb), 1)


_PLACE_TR = {0: "dudak", 1: "diş-dudak", 2: "diş eti", 3: "damak",
             4: "art damak", 5: "gırtlak", 6: "uzatma"}
_MANNER_TR = {0: "patlamalı", 1: "sızmalı", 2: "patlamalı-sızmalı",
              3: "genizsil", 4: "akıcı"}


def _describe_change(a, b) -> str:
    """Fonem değişiminin hangi ayırt edici özellikte olduğunu açıklar.
    Öğretmene sunulacak açıklanabilir çıktı için: 'b → p (ötümlülük)' gibi."""
    if a is None:
        return f"+{b} (ses eklendi)"
    if b is None:
        return f"-{a} (ses düştü)"
    fa, fb = CONSONANT_FEATURES.get(a), CONSONANT_FEATURES.get(b)
    if fa and fb:
        d = []
        if fa[0] != fb[0]:
            d.append(f"çıkış yeri: {_PLACE_TR.get(fa[0])} → {_PLACE_TR.get(fb[0])}")
        if fa[1] != fb[1]:
            d.append(f"çıkış biçimi: {_MANNER_TR.get(fa[1])} → {_MANNER_TR.get(fb[1])}")
        if fa[2] != fb[2]:
            d.append("ötümlülük")
        return f"{a} → {b} ({'; '.join(d) if d else 'aynı özellikler'})"
    va, vb = VOWEL_FEATURES.get(a), VOWEL_FEATURES.get(b)
    if va and vb:
        d = []
        if va[0] != vb[0]:
            d.append("önlük/artlık")
        if va[1] != vb[1]:
            d.append("yuvarlaklık")
        if va[2] != vb[2]:
            d.append("darlık/genişlik")
        return f"{a} → {b} ({'; '.join(d) if d else 'aynı özellikler'})"
    return f"{a} → {b}"


def mispronunciation_detail(reference_word: str, spoken_word: str, mode: str | None = None):
    """
    Telaffuz hatasının fonem-düzeyi açıklaması.
    Döndürür: {'distance', 'phoneme_ops', 'changed', 'changed_features'}
    """
    pa, pb = g2p(reference_word), g2p(spoken_word)
    raw, ops = phoneme_alignment(pa, pb, mode)
    changed = [(a, b) for (t, a, b) in ops if t != "="]
    norm = raw / max(len(pa), len(pb), 1)
    return {
        "distance": norm,
        "phoneme_ops": ops,
        "changed": changed,
        "changed_features": [_describe_change(a, b) for a, b in changed],
    }
