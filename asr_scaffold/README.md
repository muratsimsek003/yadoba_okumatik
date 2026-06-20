# HF-MTM Akustik Katman — Eğitim İskelesi (İP4)

Bu klasör, projenin özgün **HF-MTM** (Hizalama-Serbest Fonem-Düzeyi Miscue Tespit
Modeli) yönteminin **akustik** katmanının eğitim kodunu içerir. Çocuk konuşmasından
fonem-düzeyi olasılıklar üretir; çıktı, hizalama-serbest çözümleme ile bir fonem
dizisine dönüştürülerek `okumetrik` paketindeki miscue mantığını besler.

## Bu kod neden burada çalıştırılmıyor?

Eğitim üç şeyi gerektirir ve bunlar projenin gerçek İP3/İP4 fazına aittir:
1. **GPU** (wav2vec2-XLS-R ince ayarı),
2. **Önceden eğitilmiş model indirme** (Hugging Face),
3. **Etik onaylı Türkçe çocuk sesli-okuma korpusu** (İP3 çıktısı).

Kod, böyle bir ortamda çalışacak biçimde yapılandırılmıştır; sözdizimsel olarak
eksiksizdir ve `okumetrik` çekirdeğiyle aynı fonem alfabesini kullanır.

## Bileşenler

| Dosya | İşlev | İlgili İP |
|---|---|---|
| `model.py` | wav2vec2 + CTC fonem başı; `decode_free` (hizalama-serbest çözümleme); LoRA | İP4 |
| `augment.py` | Kaynak-filtre (VTLP) + hız/perde sapmasıyla çocuk-konuşması veri artırımı | İP3–İP4 |
| `data.py` | Manifest tabanlı veri seti; zayıf etiket için G2P fonem hedefi | İP3 |
| `train.py` | CTC eğitim döngüsü + aktif öğrenme (entropi) seçimi | İP4 |

## Çalıştırma (GPU'lu makinede)

```bash
pip install -r requirements.txt
# data/train.jsonl: {"audio":"wav/0001.wav","text":"...","phonemes":"k y tS ..."}
python train.py --manifest data/train.jsonl --val data/val.jsonl --lora --augment
```

## Akış (uçtan uca)

```
ses (çocuk okuması)
  -> HFMTMAcoustic.forward      (fonem posteriors)
  -> decode_free               (hizalama-serbest fonem dizisi)
  -> sözlük/G2P ile kelimeye eşleme
  -> okumetrik.analyze(referans, hipotez)   (miscue + skor + geri bildirim)
```

`augment.py` bu depoda **numpy ile çalıştırılabilir** (örnek için üst README'ye bakın);
`model/data/train` ise GPU + transformers ortamı ister.
