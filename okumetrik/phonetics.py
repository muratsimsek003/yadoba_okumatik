"""
okumetrik.phonetics — fonem dizileri arasında mesafe ve hizalama.

İki kelimenin fonem dizileri arasındaki düzenleme mesafesini hesaplar ve
hangi fonemlerin değiştiğini döndürür. Bu, bir okuma hatasının "tamamen farklı
bir kelime (substitution)" mı yoksa "telaffuz hatası (mispronunciation)" mı
olduğunu ayırt etmek için kullanılır:

  - Fonetik mesafe küçük (ör. 1 fonem farklı)  -> telaffuz hatası adayı
  - Fonetik mesafe büyük                         -> kelime değiştirme (substitution)
"""

from .g2p import g2p, is_vowel_phoneme


def _sub_cost(a: str, b: str) -> float:
    """İki fonem arasındaki yerine koyma maliyeti (0=aynı)."""
    if a == b:
        return 0.0
    av, bv = is_vowel_phoneme(a), is_vowel_phoneme(b)
    # Ünlü<->ünsüz değişimi en pahalı; aynı sınıf içi değişim daha ucuz.
    if av != bv:
        return 1.0
    return 0.6


def phoneme_alignment(pa: list[str], pb: list[str]):
    """
    Needleman-Wunsch ile iki fonem dizisini hizalar.
    Döndürür: (mesafe, ops) — ops: [('=',a,b)|('~',a,b)|('-',a,None)|('+',None,b)]
    """
    n, m = len(pa), len(pb)
    GAP = 1.0
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
            c = _sub_cost(pa[i - 1], pb[j - 1])
            diag = dp[i - 1][j - 1] + c
            up = dp[i - 1][j] + GAP
            left = dp[i][j - 1] + GAP
            best = min(diag, up, left)
            dp[i][j] = best
            bt[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    # backtrace
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


def phonetic_distance(word_a: str, word_b: str) -> float:
    """İki kelime arasındaki normalize fonetik mesafe (0..1)."""
    pa, pb = g2p(word_a), g2p(word_b)
    if not pa and not pb:
        return 0.0
    dist, _ = phoneme_alignment(pa, pb)
    return dist / max(len(pa), len(pb), 1)


def mispronunciation_detail(reference_word: str, spoken_word: str):
    """
    Telaffuz hatasının fonem-düzeyi açıklaması.
    Döndürür: {'distance', 'phoneme_ops', 'changed': [(ref_ph, spoken_ph), ...]}
    """
    pa, pb = g2p(reference_word), g2p(spoken_word)
    raw, ops = phoneme_alignment(pa, pb)
    changed = [(a, b) for (t, a, b) in ops if t != "="]
    norm = raw / max(len(pa), len(pb), 1)
    return {"distance": norm, "phoneme_ops": ops, "changed": changed}
