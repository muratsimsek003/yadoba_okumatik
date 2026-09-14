# -*- coding: utf-8 -*-
"""
Hizalama yöntemi ablasyon deneyi — dört yöntemin kontrollü karşılaştırması.

Sınanan yöntemler:
  1. classic  : klasik sabit maliyetli sözcük hizalama (yerine koyma maliyeti 1,0)
  2. ortho    : yazımsal (karakter Levenshtein) mesafe temelli hizalama
  3. class    : ünlü/ünsüz sınıfı temelli fonem mesafesi (okumetrik ilk sürüm)
  4. feature  : ayırt edici fonetik özellik temelli fonem mesafesi (okumetrik güncel)

Kullanım (depo kökünden):
    python ablation/run_ablation.py
    python ablation/run_ablation.py --reps 12 --seed 20260828

Çıktı: ablation/results/ablation_results.json + konsol tablosu.
Rastgele tohum sabittir; sonuçlar yeniden üretilebilir.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pre_feasibility"))

from okumetrik.miscue import Entry, _postprocess_repetition_selfcorrection, tokenize
from okumetrik.phonetics import phonetic_distance
from okumetrik.g2p import normalize
from synth_school_data import SCHOOL_SENTENCES

CLASSES = ["substitution", "mispronunciation", "omission"]
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Kontrollü hata korpusu üretimi
# ---------------------------------------------------------------------------
VOWEL_NEIGH = {"a": "ae", "e": "ai", "ı": "iu", "i": "ıe",
               "o": "öu", "ö": "oü", "u": "üo", "ü": "ui"}
CONS_PAIR = {"b": "p", "p": "b", "d": "t", "t": "d", "c": "ç", "ç": "c",
             "g": "k", "k": "g", "z": "s", "f": "v", "v": "f", "ş": "s", "s": "ş"}

VOCAB = sorted({normalize(w) for s in SCHOOL_SENTENCES
                for w in tokenize(s) if len(normalize(w)) >= 3})


def make_mispron(w: str):
    """Tek fonem düzeyinde sapma: ünlü değişimi, ünsüz ötümlülük değişimi, ses düşmesi.
    Üretilen biçim sözlükte bulunmamalıdır (aksi hâlde sözcük değiştirme olurdu)."""
    w = normalize(w)
    cand = []
    for i, ch in enumerate(w):
        for r in VOWEL_NEIGH.get(ch, ""):
            cand.append(w[:i] + r + w[i + 1:])
        if ch in CONS_PAIR:
            cand.append(w[:i] + CONS_PAIR[ch] + w[i + 1:])
    if len(w) > 4:
        cand.append(w[:-1])            # son ses düşmesi
        cand.append(w[:2] + w[3:])     # iç ses düşmesi
    cand = [c for c in cand if c and c != w and c not in VOCAB]
    return random.choice(cand) if cand else None


def make_subst(w: str, hard: bool):
    """Gerçek sözcükle değiştirme. hard=True ise fonetik olarak en yakın 5 sözcükten biri."""
    w = normalize(w)
    others = [v for v in VOCAB if v != w]
    if not others:
        return None
    if hard:
        return random.choice(sorted(others, key=lambda v: phonetic_distance(w, v))[:5])
    return random.choice(others)


def build_case(sentence: str, hard: bool):
    ref = tokenize(sentence)
    if len(ref) < 4:
        return None
    hyp, gold, plan = [], {}, {}
    idxs = list(range(len(ref)))
    random.shuffle(idxs)
    for k, idx in enumerate(idxs[:max(2, len(ref) // 3)]):
        plan[idx] = CLASSES[k % 3] if CLASSES[k % 3] != "omission" else "omission"
        plan[idx] = ["mispronunciation", "substitution", "omission"][k % 3]
    for i, w in enumerate(ref):
        ev = plan.get(i)
        if ev == "mispronunciation":
            m = make_mispron(w)
            hyp.append(m or w); gold[i] = "mispronunciation" if m else "correct"
        elif ev == "substitution":
            s = make_subst(w, hard)
            hyp.append(s or w); gold[i] = "substitution" if s else "correct"
        elif ev == "omission":
            gold[i] = "omission"
        else:
            hyp.append(w); gold[i] = "correct"
    return ref, hyp, gold


# ---------------------------------------------------------------------------
# Hizalama yöntemleri
# ---------------------------------------------------------------------------
def _nw(ref, hyp, cost_fn, gap=1.0, locality=0.0):
    """Ortak Needleman-Wunsch iskeleti. cost_fn(r,h) -> (maliyet, etiket)."""
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap; bt[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap; bt[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost_fn(ref[i - 1], hyp[j - 1])[0]
            if locality and n and m:
                c += locality * abs(i / n - j / m)
            diag, up, left = dp[i-1][j-1] + c, dp[i-1][j] + gap, dp[i][j-1] + gap
            best = min(diag, up, left)
            dp[i][j] = best
            bt[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    entries, i, j = [], n, m
    while i > 0 or j > 0:
        mv = bt[i][j]
        if mv == "diag":
            r, h = ref[i-1], hyp[j-1]
            entries.append(Entry(cost_fn(r, h)[1], r, h, i-1, j-1, {})); i, j = i-1, j-1
        elif mv == "up":
            entries.append(Entry("omission", ref[i-1], None, i-1, None)); i -= 1
        else:
            entries.append(Entry("insertion", None, hyp[j-1], None, j-1)); j -= 1
    entries.reverse()
    return _postprocess_repetition_selfcorrection(entries)


def align_classic(ref, hyp):
    """Baz çizgi 1: sabit maliyet. Telaffuz sapması sınıfını üretemez."""
    def cost(r, h):
        return (0.0, "correct") if r.lower() == h.lower() else (1.0, "substitution")
    return _nw(ref, hyp, cost)


def _lev(a, b):
    n, m = len(a), len(b)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (a[i-1] != b[j-1]))
    return d[n][m]


def make_align_ortho(thr):
    """Baz çizgi 2: yazımsal mesafe (literatürdeki rakip yaklaşım)."""
    def cost(r, h):
        if r.lower() == h.lower():
            return -0.1, "correct"
        d = _lev(r.lower(), h.lower()) / max(len(r), len(h), 1)
        return (0.4, "mispronunciation") if d <= thr else (1.0, "substitution")
    return lambda ref, hyp: _nw(ref, hyp, cost, locality=0.05)


def make_align_phoneme(mode, thr):
    """Önerilen yöntem: fonem mesafesi temelli (mode = 'class' | 'feature')."""
    def cost(r, h):
        if r.lower() == h.lower():
            return -0.1, "correct"
        d = phonetic_distance(r, h, mode=mode)
        return (0.4, "mispronunciation") if d <= thr else (1.0, "substitution")
    return lambda ref, hyp: _nw(ref, hyp, cost, locality=0.05)


# ---------------------------------------------------------------------------
# Değerlendirme
# ---------------------------------------------------------------------------
def macro_f1(gold, pred):
    per = {}
    for c in CLASSES:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        pr = tp/(tp+fp) if tp+fp else 0.0
        rc = tp/(tp+fn) if tp+fn else 0.0
        per[c] = {"precision": round(pr, 4), "recall": round(rc, 4),
                  "f1": round(2*pr*rc/(pr+rc), 4) if pr+rc else 0.0, "n": tp+fn}
    return round(sum(per[c]["f1"] for c in CLASSES)/len(CLASSES), 4), per


def evaluate(align_fn, hard, reps, seed):
    random.seed(seed)
    G, P = [], []
    for _ in range(reps):
        for s in SCHOOL_SENTENCES:
            case = build_case(s, hard)
            if not case:
                continue
            ref, hyp, gold = case
            pred = {e.ref_idx: e.status for e in align_fn(ref, hyp) if e.ref_idx is not None}
            for i, g in gold.items():
                if g in CLASSES or pred.get(i, "correct") in CLASSES:
                    G.append(g); P.append(pred.get(i, "correct"))
    mf1, per = macro_f1(G, P)
    return mf1, per, len(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--sweep", action="store_true",
                    help="eşik taraması yap (yavaş)")
    a = ap.parse_args()

    methods = {
        "classic": align_classic,
        "ortho (esik 0.34)": make_align_ortho(0.34),
        "class  (esik 0.34)": make_align_phoneme("class", 0.34),
        "feature(esik 0.20)": make_align_phoneme("feature", 0.20),
    }
    out = {"seed": a.seed, "reps": a.reps, "vocab_size": len(VOCAB),
           "n_sentences": len(SCHOOL_SENTENCES), "conditions": {}}

    for label, hard in [("kolay_rastgele_sozcuk", False),
                        ("zorlayici_fonetik_komsu", True)]:
        print(f"\n=== {label} ===")
        out["conditions"][label] = {}
        base = None
        for name, fn in methods.items():
            mf1, per, n = evaluate(fn, hard, a.reps, a.seed)
            if base is None:
                base = mf1
            print(f"  {name:20s} Macro-F1 = {mf1:.4f}   (olay: {n})")
            out["conditions"][label][name] = {"macro_f1": mf1, "per_class": per, "n_events": n}
        cls = out["conditions"][label]
        d_classic = cls["feature(esik 0.20)"]["macro_f1"] - cls["classic"]["macro_f1"]
        d_ortho = cls["feature(esik 0.20)"]["macro_f1"] - cls["ortho (esik 0.34)"]["macro_f1"]
        print(f"  --> klasik baz cizgiye gore kazanc  : {d_classic:+.4f}")
        print(f"  --> yazimsal baz cizgiye gore kazanc: {d_ortho:+.4f}")
        cls["_delta_vs_classic"] = round(d_classic, 4)
        cls["_delta_vs_ortho"] = round(d_ortho, 4)

    if a.sweep:
        print("\n=== esik taramasi (zorlayici kosul) ===")
        out["sweep"] = {}
        for mode, thrs in [("ortho", [0.20, 0.25, 0.30, 0.34, 0.40]),
                           ("class", [0.25, 0.30, 0.34, 0.40]),
                           ("feature", [0.15, 0.20, 0.25, 0.30, 0.35])]:
            out["sweep"][mode] = {}
            for t in thrs:
                fn = make_align_ortho(t) if mode == "ortho" else make_align_phoneme(mode, t)
                mf1, _, _ = evaluate(fn, True, a.reps, a.seed)
                print(f"  {mode:8s} esik {t:.2f} -> {mf1:.4f}")
                out["sweep"][mode][str(t)] = mf1

    path = RESULTS_DIR / "ablation_results.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSonuclar yazildi: {path}")


if __name__ == "__main__":
    main()
