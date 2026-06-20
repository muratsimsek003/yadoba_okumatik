# Ön Fizibilite — Türkçe ASR Fine-tune + WER (AGY100 B.3)

Bu klasör, başvuruya konacak **somut WER ön fizibilite kanıtını** üretir.

## Neden burada (Anthropic ortamında) gerçek sayı üretilemedi?

Bu ortamda iki sert engel var ve dürüstçe belirtilmelidir:
- **huggingface.co engelli** (`x-deny-reason: host_not_allowed`) → hazır model ve
  veri seti indirilemiyor.
- **GPU yok** (`torch.cuda.is_available() == False`).

Bu yüzden gerçek Türkçe WER aşağıdaki betikle **Colab/GPU'da** üretilir. Pipeline'ın
doğru çalıştığı ise `selftest_offline.py` ile ağ/GPU'suz **kanıtlanmıştır**
(sentetik veride WER %100 → %0; CTC eğitimi + çözümleme + WER hesabı doğru).

## İçindekiler

| Dosya | İşlev | Çalışma yeri |
|---|---|---|
| `selftest_offline.py` | Pipeline doğrulama (sentetik, gerçek CTC eğitimi + WER) | ✅ Her yerde (CPU) |
| `finetune_whisper_tr.py` | GERÇEK: Whisper + Türkçe veri seti fine-tune + WER | GPU/Colab |
| `requirements.txt` | Bağımlılıklar | — |

## En hızlı PoC: fine-tune'suz (zero-shot) WER

En düşük efor: hazır Whisper'ın Türkçe WER'ini ölç (eğitim yok, dakikalar sürer):

```bash
pip install -r requirements.txt
python finetune_whisper_tr.py --dataset fleurs --model openai/whisper-small --baseline_only
```

## Tam PoC: fine-tune + WER (önce/sonra)

```bash
python finetune_whisper_tr.py \
    --dataset fleurs --model openai/whisper-small \
    --lora --epochs 3 --max_train 1500 --max_eval 400
```

Çıktı:
```
Zero-shot WER     : %XX.XX
Fine-tune sonrası : %YY.YY
```

## Veri seti seçenekleri (hazır)

- **`--dataset fleurs`** → `google/fleurs` (config `tr_tr`). **Açık**, token gerektirmez. PoC için önerilir (~3-4 saat ses).
- **`--dataset common_voice`** → `mozilla-foundation/common_voice_17_0` (`tr`).
  Daha büyük; HF token + şartların kabulü gerekir (`huggingface-cli login`).

## Beklenen WER aralıkları (literatür — başvuruya referans)

Türkçe ASR düşük-kaynaklı kabul edilir; yayınlanmış değerler:
- Whisper + LoRA, temiz Türkçe: **WER ~%4.3–14.2** (Koçak & Ulaş, 2024, Electronics 13(21):4227).
- Zorlu/konuşma diline yakın (Babel): **WER ~%38.9 / CER ~%22.2**.

> **Stratejik not (önemli):** Bu ön fizibilite, temiz Türkçe ASR'nin zaten düşük WER
> verdiğini (yani "çözülmüş" olduğunu) **doğrular** — bu, reddedilen başvurunun "≤%5
> WER" hedefinin neden yenilik sayılmadığını gösterir. YADOBA 2.0'da WER **baş gösterge
> değildir**; baş gösterge, çözülmemiş olan **çocuk sesli-okumasında miscue F1**'dir
> (bkz. ana README, `okumetrik`). WER yalnızca akustik bileşenin destekleyici metriğidir.

## Çocuk konuşmasına genişletme (gerçek proje, İP3/İP4)

PoC açık yetişkin verisiyle yapılır. Gerçek projede:
1. İP3 çocuk korpusu toplanır (etik onaylı).
2. `asr_scaffold/augment.py` (kaynak-filtre/VTLP) ile yetişkin→çocuk uyarlaması.
3. `--dataset` yerine çocuk korpus manifesti bağlanır; az-etiketli ayarda WER ve
   miscue F1 birlikte raporlanır.

## Colab hızlı başlangıç

```python
# Colab hücresi
!git clone <bu_repo>  # veya zip'i yükleyip açın
%cd yadoba-okumetrik/pre_feasibility
!pip install -q -r requirements.txt
!python finetune_whisper_tr.py --dataset fleurs --model openai/whisper-small --lora --epochs 3
```
