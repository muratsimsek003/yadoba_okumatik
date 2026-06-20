#!/usr/bin/env python3
"""
OKUMETRİK demo — gerçekçi bir Türkçe çocuk okuma senaryosunu analiz eder,
terminale rapor basar ve tek-dosya bir HTML rapor görselleştiricisi üretir.

Gerçek sistemde 'hypothesis' HF-MTM/ASR çıktısıdır. Burada, eğitilmiş model
ve ses verisi gerektirmeden çekirdeği göstermek için tanıma çıktısı simüle
edilmiştir (kelime + zaman damgası).
"""
import os, sys, json, html
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from okumetrik import analyze, evaluate

# Çocuğun okuması beklenen metin (telifsiz, basit)
REFERENCE = (
    "Küçük kırmızı balık derin denizde yüzüyordu. "
    "Annesini aramak için mercan kayalıklarının arasında dolaştı. "
    "Sonunda sıcak ve aydınlık sulara ulaştı."
)

# Simüle edilmiş tanıma çıktısı (çocuğun gerçek okuması gibi, hatalı):
#  - kırmızı -> kırmız (telaffuz), balık -> kuş (değiştirme)
#  - "derin" atlandı, "çok" eklendi
#  - "annesini" -> "annesi" (telaffuz), "mercan" tekrarı
#  - "kayalıklarının" -> "kayalıkların" (telaffuz)
#  - "aydınlık" önce "aydılık" denenip düzeltildi (kendi kendine düzeltme)
HYP_WORDS = [
    "küçük","kırmız","kuş","denizde","çok","yüzüyordu",
    "annesi","aramak","için","mercan","mercan","kayalıkların","arasında","dolaştı",
    "sonunda","sıcak","ve","aydılık","aydınlık","sulara","ulaştı",
]
# basit eşit aralıklı zaman damgaları (toplam ~38 sn okuma)
def mk_hyp(words, total=38.0):
    step = total / len(words)
    return [{"word": w, "start": round(i*step,2), "end": round((i+1)*step,2)} for i,w in enumerate(words)]

HYP = mk_hyp(HYP_WORDS)

# Uzman (altın) etiketi: hatalı referans kelime indeksleri (Ölçüt 1 değerlendirmesi)
# ref indeksleri: 0 küçük,1 kırmızı,2 balık,3 derin,4 denizde,5 yüzüyordu,
# 6 annesini,7 aramak,8 için,9 mercan,10 kayalıklarının,11 arasında,12 dolaştı,
# 13 sonunda,14 sıcak,15 ve,16 aydınlık,17 sulara,18 ulaştı
GOLD_MISCUES = {1, 2, 3, 6, 10}   # telaffuz/değiştirme/atlama olanlar


COLOR = {
    "correct":"\033[92m","substitution":"\033[91m","mispronunciation":"\033[93m",
    "omission":"\033[95m","insertion":"\033[96m","repetition":"\033[94m",
    "self_correction":"\033[90m",
}
RESET="\033[0m"

def print_report(report, entries):
    print("="*64)
    print("OKUMETRİK — OKUMA DEĞERLENDİRME RAPORU (PoC)")
    print("="*64)
    print("\nHizalanmış okuma (renk = miscue türü):")
    line=[]
    for e in entries:
        w = e.hyp or e.ref or "?"
        c = COLOR.get(e.status,"")
        tag = "" if e.status=="correct" else f"[{e.status[:4]}]"
        line.append(f"{c}{w}{tag}{RESET}")
    print("  "+" ".join(line))
    m = report["metrics"]
    print("\nMetrikler:")
    print(f"  Toplam kelime      : {m['total_words']}")
    print(f"  Doğru okunan       : {m['words_correct']}")
    print(f"  Doğruluk           : %{m['accuracy']*100:.1f}")
    print(f"  WCPM               : {m['wcpm']}")
    print(f"  Okuma hızı (wpm)   : {m['rate_wpm']}")
    print(f"  Akıcılık skoru     : {m['fluency_score']}/100")
    print(f"  Miscue sayımı      : {m['miscue_counts']}")
    print(f"\n  {report['overall_feedback']}")
    print("\nKelime bazında geri bildirim:")
    for f in report["word_feedback"]:
        print("  - "+f)
    res = evaluate(entries, GOLD_MISCUES)
    print("\nÖlçüt 1 (uzman altın etikete karşı miscue tespiti):")
    print(f"  precision={res['precision']}  recall={res['recall']}  F1={res['f1']}  "
          f"(tp={res['tp']} fp={res['fp']} fn={res['fn']})")
    return res


HTML_CSS = """
body{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:24px auto;color:#222;padding:0 16px}
h1{color:#1F4E79;border-bottom:3px solid #1F4E79;padding-bottom:6px;font-size:22px}
h2{color:#2E5496;font-size:17px;margin-top:24px}
.reading{line-height:2.1;font-size:18px;background:#fafafa;border:1px solid #e3e3e3;padding:14px;border-radius:8px}
.tok{padding:2px 5px;border-radius:5px;margin:1px}
.correct{color:#1a7f37}
.substitution{background:#ffd6d6;color:#b30000;text-decoration:line-through}
.mispronunciation{background:#fff2c2;color:#8a6d00}
.omission{background:#eee;color:#999;text-decoration:underline dotted}
.insertion{background:#d6ecff;color:#0057b3;font-style:italic}
.repetition{background:#e3dbff;color:#5b2bcc}
.self_correction{background:#e9ffe9;color:#2a8a2a}
.metrics{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
.card{flex:1;min-width:130px;background:#1F4E79;color:#fff;border-radius:8px;padding:10px 12px}
.card .v{font-size:22px;font-weight:bold}
.card .l{font-size:12px;opacity:.85}
ul{line-height:1.6}
.legend span{margin-right:12px;font-size:13px}
.tag{font-size:10px;vertical-align:super;opacity:.7}
.note{color:#666;font-size:12px;font-style:italic}
"""

def to_html(report, entries, res, path):
    toks=[]
    for e in entries:
        w = html.escape(e.hyp or e.ref or "?")
        tag = "" if e.status=="correct" else f"<span class='tag'>{e.status[:4]}</span>"
        toks.append(f"<span class='tok {e.status}'>{w}{tag}</span>")
    m=report["metrics"]
    cards=[("Doğruluk",f"%{m['accuracy']*100:.0f}"),("WCPM",m['wcpm']),
           ("Akıcılık",f"{m['fluency_score']}"),("Hata",m['total_errors']),
           ("Miscue F1",res['f1'])]
    card_html="".join(f"<div class='card'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for l,v in cards)
    fb="".join(f"<li>{html.escape(x)}</li>" for x in report["word_feedback"])
    legend=("<div class='legend'><span class='correct'>■ doğru</span>"
            "<span style='color:#b30000'>■ değiştirme</span>"
            "<span style='color:#8a6d00'>■ telaffuz</span>"
            "<span style='color:#999'>■ atlama</span>"
            "<span style='color:#0057b3'>■ ekleme</span>"
            "<span style='color:#5b2bcc'>■ tekrar</span>"
            "<span style='color:#2a8a2a'>■ kendi kendine düzeltme</span></div>")
    doc=f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>OKUMETRİK Raporu</title><style>{HTML_CSS}</style></head><body>
<h1>OKUMETRİK — Okuma Değerlendirme Raporu</h1>
<p class="note">YADOBA 2.0 çekirdeği (PoC). Tanıma çıktısı bu demoda simüle edilmiştir; üretimde HF-MTM/ASR'den gelir.</p>
<h2>Genel Değerlendirme</h2><p>{html.escape(report['overall_feedback'])}</p>
<div class="metrics">{card_html}</div>
<h2>Hizalanmış Okuma</h2>{legend}
<div class="reading">{' '.join(toks)}</div>
<h2>Kelime Bazında Geri Bildirim</h2><ul>{fb}</ul>
<h2>Ölçüt 1 — Uzman Etikete Karşı Doğrulama</h2>
<p>precision = {res['precision']} &nbsp; recall = {res['recall']} &nbsp; <b>F1 = {res['f1']}</b>
(tp={res['tp']}, fp={res['fp']}, fn={res['fn']})</p>
</body></html>"""
    with open(path,"w",encoding="utf-8") as f: f.write(doc)


if __name__ == "__main__":
    report, entries = analyze(REFERENCE, HYP)
    res = print_report(report, entries)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir,"report.json"),"w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    to_html(report, entries, res, os.path.join(out_dir,"report.html"))
    print(f"\nÜretildi: report.json, report.html")
