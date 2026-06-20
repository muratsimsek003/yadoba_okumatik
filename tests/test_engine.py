import os, sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okumetrik import g2p, phonetic_distance, analyze, evaluate
from okumetrik.miscue import align, tokenize


def test_g2p_basic():
    assert g2p("balık") == ["b", "a", "l", "I", "k"]
    assert g2p("şeker") == ["S", "E", "k", "E", "r"]
    assert g2p("çocuk") == ["tS", "o", "dZ", "u", "k"]
    # yumuşak g önceki ünlüyü uzatır
    assert ":" in g2p("dağ")


def test_phonetic_distance():
    assert phonetic_distance("kırmızı", "kırmızı") == 0.0
    # tek fonem düşmesi -> küçük mesafe
    assert phonetic_distance("kırmızı", "kırmız") < 0.34
    # tamamen farklı kelime -> büyük mesafe
    assert phonetic_distance("balık", "kuş") > 0.5


def test_miscue_types():
    ref = "küçük kırmızı balık derin denizde mutlu mutlu yüzüyordu"
    hyp = "küçük kırmız kuş denizde çok mutlu mutlu mutlu yüzüyordu"
    report, entries = analyze(ref, hyp)
    by_idx = {e.ref_idx: e.status for e in entries if e.ref_idx is not None}
    assert by_idx[0] == "correct"            # küçük
    assert by_idx[1] == "mispronunciation"   # kırmızı -> kırmız
    assert by_idx[2] == "substitution"       # balık -> kuş
    assert by_idx[3] == "omission"           # derin atlandı
    assert by_idx[4] == "correct"            # denizde
    statuses = [e.status for e in entries]
    assert "insertion" in statuses           # çok
    assert "repetition" in statuses          # 3. mutlu


def test_self_correction():
    ref = "büyük balık"
    hyp = "büyük balok balık"   # balok (yanlış) -> balık (düzeltme)
    report, entries = analyze(ref, hyp)
    statuses = [e.status for e in entries]
    assert "self_correction" in statuses


def test_f1_against_gold():
    ref = "küçük kırmızı balık derin denizde mutlu mutlu yüzüyordu"
    hyp = "küçük kırmız kuş denizde çok mutlu mutlu mutlu yüzüyordu"
    report, entries = analyze(ref, hyp)
    gold = {1, 2, 3}  # uzman: kırmızı(telaffuz), balık(değiştirme), derin(atlama)
    res = evaluate(entries, gold)
    assert res["f1"] == 1.0, res


def test_scoring_accuracy():
    ref = "ali topu at"
    hyp = "ali topu at"
    report, _ = analyze(ref, hyp, audio_duration_sec=3.0)
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["wcpm"] == 60.0  # 3 kelime / 0.05 dk


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); ok += 1
        except Exception:
            print(f"FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{ok}/{len(fns)} test geçti")
    sys.exit(0 if ok == len(fns) else 1)
