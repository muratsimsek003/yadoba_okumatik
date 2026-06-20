"""
okumetrik.miscue — referans metin ile okuma (tanıma) çıktısının hizalanması
ve yanlış okuma (miscue) türlerinin sınıflandırılması.

Tespit edilen miscue türleri:
  correct          : doğru okunan kelime
  substitution     : kelime tamamen farklı bir kelimeyle değiştirilmiş
  mispronunciation : kelime fonem-düzeyinde yanlış telaffuz edilmiş (yakın)
  omission         : kelime atlanmış (okunmamış)
  insertion        : metinde olmayan kelime eklenmiş
  repetition       : kelime/kelimeler tekrar okunmuş
  self_correction  : önce yanlış okunup hemen ardından düzeltilmiş

Hizalama, sözcük düzeyinde Needleman-Wunsch ile yapılır; yerine koyma maliyeti
fonetik mesafeyle ağırlıklandırılır (telaffuz hatası vs. kelime değiştirme ayrımı).
"""

from dataclasses import dataclass, field
from .phonetics import phonetic_distance, mispronunciation_detail

MISPRON_THRESHOLD = 0.34   # fonetik mesafe bu eşiğin altındaysa telaffuz hatası
_PUNC = ".,;:!?\"'()[]{}—–-…«»“”’"


def tokenize(text: str) -> list[str]:
    out = []
    for raw in text.replace("\n", " ").split():
        w = raw.strip(_PUNC)
        if w:
            out.append(w)
    return out


@dataclass
class Entry:
    status: str
    ref: str | None = None
    hyp: str | None = None
    ref_idx: int | None = None
    hyp_idx: int | None = None
    detail: dict = field(default_factory=dict)


def _word_sub_cost(ref: str, hyp: str) -> tuple[float, str]:
    # Tam eşleşmeye küçük bir ödül (negatif maliyet) verilir; böylece eşit
    # toplam maliyetli alternatifler arasında aynı kelimeleri hizalayan,
    # pedagojik olarak doğru hizalama tercih edilir (hizalama belirsizliği çözümü).
    if ref.lower() == hyp.lower():
        return -0.1, "correct"
    d = phonetic_distance(ref, hyp)
    if d <= MISPRON_THRESHOLD:
        return 0.4, "mispronunciation"
    return 1.0, "substitution"


def align(ref_tokens: list[str], hyp_tokens: list[str]) -> list[Entry]:
    n, m = len(ref_tokens), len(hyp_tokens)
    GAP = 1.0
    LOC = 0.05  # yerellik tie-breaker: benzer göreli konumdaki kelimeleri hizala
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP; bt[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * GAP; bt[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, _ = _word_sub_cost(ref_tokens[i - 1], hyp_tokens[j - 1])
            if n and m:
                c += LOC * abs(i / n - j / m)
            diag = dp[i - 1][j - 1] + c
            up = dp[i - 1][j] + GAP
            left = dp[i][j - 1] + GAP
            best = min(diag, up, left)
            dp[i][j] = best
            bt[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    entries: list[Entry] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "diag":
            ref, hyp = ref_tokens[i - 1], hyp_tokens[j - 1]
            _, status = _word_sub_cost(ref, hyp)
            detail = {}
            if status in ("mispronunciation", "substitution"):
                detail = mispronunciation_detail(ref, hyp)
            entries.append(Entry(status, ref, hyp, i - 1, j - 1, detail))
            i, j = i - 1, j - 1
        elif move == "up":
            entries.append(Entry("omission", ref_tokens[i - 1], None, i - 1, None))
            i -= 1
        else:
            entries.append(Entry("insertion", None, hyp_tokens[j - 1], None, j - 1))
            j -= 1
    entries.reverse()
    return _postprocess_repetition_selfcorrection(entries)


def _postprocess_repetition_selfcorrection(entries: list[Entry]) -> list[Entry]:
    """
    İki davranışı yakalar:
      - Tekrar (repetition): eklenen kelime, komşu doğru okunan referans kelimesiyle
        aynıysa, 'insertion' yerine 'repetition' olarak işaretlenir.
      - Kendi kendine düzeltme (self_correction): bir kelime yanlış okunup
        (substitution/mispronunciation) hemen ardından doğru kelime eklenmişse,
        ilk hata 'self_correction' olarak işaretlenir (doğruluk cezası verilmez).
    """
    out = list(entries)

    # self-correction A: [hata(ref=X)] + [insertion(hyp ~ X, X'ten farklı)]
    for k in range(len(out) - 1):
        e, nxt = out[k], out[k + 1]
        if e.status in ("substitution", "mispronunciation") and nxt.status == "insertion":
            if e.ref and nxt.hyp and phonetic_distance(e.ref, nxt.hyp) <= MISPRON_THRESHOLD:
                e.status = "self_correction"
                e.detail = {"corrected_to": nxt.hyp, "first_attempt": e.hyp}
                nxt.status = "_consumed"
    out = [e for e in out if e.status != "_consumed"]

    # repetition: insertion'ın yüzeyi, komşu doğru kelimeyle BİREBİR aynı
    for k, e in enumerate(out):
        if e.status != "insertion" or not e.hyp:
            continue
        prev_ok = out[k - 1] if k > 0 else None
        next_ok = out[k + 1] if k + 1 < len(out) else None
        for neigh in (prev_ok, next_ok):
            if neigh and neigh.status in ("correct", "mispronunciation") and neigh.ref \
               and neigh.ref.lower() == e.hyp.lower():
                e.status = "repetition"
                break

    # self-correction B: [insertion(hyp=Y)] + [correct(ref=X, hyp=X)], Y~X ama Y!=X
    for k in range(len(out) - 1):
        e, nxt = out[k], out[k + 1]
        if e.status == "insertion" and e.hyp and nxt.status in ("correct", "mispronunciation") \
           and nxt.ref and e.hyp.lower() != nxt.ref.lower():
            d = phonetic_distance(e.hyp, nxt.ref)
            if 0 < d <= MISPRON_THRESHOLD:
                e.status = "self_correction"   # ref=None; kredi nxt'in correct'inden
                e.detail = {"first_attempt": e.hyp, "corrected_to": nxt.ref}
    return out
