# YADOBA 2.0 — OKUMETRİK Okuma Değerlendirme Motoru

Türkçe çocuk **sesli okumasında** yanlış okuma (miscue) ve fonem-düzeyi telaffuz
hatalarını otomatik tespit eden, okuma akıcılığı skorları ve açıklanabilir Türkçe
geri bildirim üreten sistemin **çekirdek yazılımı**. TÜBİTAK 1501 başvurusunun
(YADOBA 2.0) teknik karşılığıdır.

> **Dürüst kapsam:** 15 aylık projenin tamamı (eğitilmiş HF-MTM modeli + etik
> onaylı çocuk ses korpusu + GPU eğitimi) tek seferde teslim edilemez; bunlar
> gerçek **veri ve donanım** gerektiren İP3/İP4 fazlarıdır. Bu depo, **bugün
> çalışan ve test edilen** çekirdeği (OKUMETRİK motoru + Türkçe G2P) ve akustik
> modelin (HF-MTM) **eğitim iskelesini** içerir. Motor, herhangi bir
> `(referans, tanıma çıktısı)` çifti üzerinde hemen çalışır ve doğrudan başvuruya
> eklenecek **PoC** olarak kullanılabilir.

## Ne çalışır (bu ortamda, bağımlılıksız)

```bash
python3 tests/test_engine.py      # 6/6 birim testi
python3 examples/run_demo.py      # örnek analiz + report.html / report.json üretir
```

`okumetrik` paketi **saf Python**'dur (harici bağımlılık yok): hizalama, miscue
sınıflandırma, skorlama ve Türkçe G2P. Demo, gerçekçi bir Türkçe okuma senaryosunu
analiz eder; tanıma çıktısı (gerçek sistemde HF-MTM'den gelir) burada simüle edilir.

## Mimari

```
ses (çocuk okuması)
   │  [İP4: asr_scaffold — GPU + veri ile eğitilir]
   ▼
HF-MTM akustik model  →  hizalama-serbest fonem dizisi  →  kelimeye eşleme
   │
   ▼  [İP5: okumetrik — bu depoda çalışır]
okumetrik.analyze(referans_metin, tanıma_çıktısı)
   ├─ miscue.align        : sözcük + fonem hizalama (yerellik tie-breaker'lı NW)
   ├─ phonetics/g2p       : telaffuz hatası vs kelime değiştirme ayrımı
   ├─ scoring             : WCPM, doğruluk, akıcılık, miscue F1 (Ölçüt 1/2)
   └─ feedback            : açıklanabilir Türkçe geri bildirim
```

## Tespit edilen miscue türleri

`correct`, `substitution` (kelime değiştirme), `mispronunciation` (fonem-düzeyi
telaffuz hatası), `omission` (atlama), `insertion` (ekleme), `repetition` (tekrar),
`self_correction` (kendi kendine düzeltme).

## Dosya yapısı ve İP eşleşmesi

| Yol | İçerik | İlgili İP | Durum |
|---|---|---|---|
| `okumetrik/g2p.py` | Türkçe kural-tabanlı grapheme-to-phoneme | İP4 | ✅ Çalışır |
| `okumetrik/phonetics.py` | Fonem-düzeyi mesafe ve hizalama (MDD ayrımı) | İP4 | ✅ Çalışır |
| `okumetrik/miscue.py` | Sözcük hizalama + miscue sınıflandırma | İP5 | ✅ Çalışır |
| `okumetrik/scoring.py` | WCPM/doğruluk/akıcılık + Ölçüt 1/2 değerlendirme | İP5 | ✅ Çalışır |
| `okumetrik/feedback.py` | Açıklanabilir Türkçe geri bildirim | İP5 | ✅ Çalışır |
| `okumetrik/pipeline.py` | Üst düzey `analyze()` / `evaluate()` API | İP5 | ✅ Çalışır |
| `examples/run_demo.py` | Demo + HTML/JSON rapor üretimi | İP6 | ✅ Çalışır |
| `tests/test_engine.py` | Birim testleri (6/6) | — | ✅ Geçer |
| `asr_scaffold/` | HF-MTM akustik model eğitim iskelesi | İP3–İP4 | ⚙ GPU+veri ile |

## Başarı ölçütleriyle bağ (AGY100 B.3)

- **Ölçüt 1 (miscue F1 ≥ 0,80):** `scoring.evaluate_miscues` uzman altın etikete
  karşı precision/recall/F1 hesaplar. Demo örneğinde sentetik altın etikete karşı
  F1 = 1,0 (sentetik veridir; gerçek değer İP4/İP7'de korpus üzerinde ölçülür).
- **Ölçüt 2 (telaffuz precision/recall):** `mispronunciation` etiketleri + fonem
  farkları (`phonetics.mispronunciation_detail`).
- **Ölçüt 3 (uzman korelasyonu r ≥ 0,80):** `scoring.summarize` skorları üretir;
  korelasyon, uzman puanlı veriyle (İP5) ölçülür.
- **Ölçüt 4 (az-etiketli kayıp ≤ %5):** `asr_scaffold/train.py` aktif öğrenme +
  augment ile öğrenme eğrisi deneyini destekler.
- **Ölçüt 5 (gecikme ≤ 1,5 sn):** üretim demonstratöründe (İP6) ölçülür.

## Sonraki adımlar (gerçek proje)

1. İP3: Etik onaylı korpusun toplanması ve uzman etiketlemesi → `data.py` manifesti.
2. İP4: `asr_scaffold` ile HF-MTM eğitimi (GPU); `decode_free` çıktısının
   `okumetrik` ile entegrasyonu.
3. İP5: uzman puanıyla kalibrasyon ve r korelasyonu.
4. İP6: KVKK-uyumlu web/mobil demonstratör (`report.html` bu arayüzün tohumudur).

## Lisans / Etik

Çocuk ses verisi KVKK + Etik Kurul kapsamında işlenmelidir (bkz. AGY100 İP2).
Bu depo kod ve metodolojidir; herhangi bir kişisel veri içermez.
