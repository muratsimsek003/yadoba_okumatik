# Teknik Rapor — YADOBA 2.0 / OKUMETRİK ASR Geliştirme Süreci

**Tarih:** 2026-06-19  
**Proje:** YADOBA 2.0 — Türkçe Çocuk Sesli Okuma Değerlendirme Motoru  
**Kapsam:** Offline pipeline doğrulama → Gerçek Whisper fine-tune → Domain-specific çocuk ASR altyapısı  
**Hazırlayan:** Claude Code (Anthropic) + Proje Ekibi

---

## İçindekiler

1. [Proje Genel Bakışı](#1-proje-genel-bakışı)
2. [Donanım ve Ortam](#2-donanım-ve-ortam)
3. [Aşama 1 — Offline Pipeline Doğrulama](#3-aşama-1--offline-pipeline-doğrulama)
4. [Aşama 2 — Gerçek Whisper Fine-tune (FLEURS)](#4-aşama-2--gerçek-whisper-fine-tune-fleurs)
5. [Karşılaşılan Sorunlar ve Çözümler](#5-karşılaşılan-sorunlar-ve-çözümler)
6. [Fine-tune Sonuçları](#6-fine-tune-sonuçları)
7. [Domain-Specific Çocuk ASR Planı](#7-domain-specific-çocuk-asr-planı)
8. [WER %5 Hedefine Yol Haritası](#8-wer-5-hedefine-yol-haritası)
9. [Oluşturulan Dosyalar](#9-oluşturulan-dosyalar)
10. [Referanslar](#10-referanslar)

---

## 1. Proje Genel Bakışı

**YADOBA 2.0 / OKUMETRİK**, Türkçe çocuk sesli okumasında otomatik değerlendirme yapan bir sistemdir. TÜBİTAK 1501 kapsamında geliştirilen bu sistem:

- **Miscue tespiti:** substitution, omission, insertion, repetition, self-correction, mispronunciation
- **Akıcılık ölçümü:** WCPM (Words Correct Per Minute), doğruluk skoru
- **Fonem analizi:** Türkçe G2P ile telaffuz hatası tespiti
- **Geri bildirim:** Açıklanabilir Türkçe pedagojik geri bildirim

### Sistem Mimarisi

```
Çocuk Sesi (mikrofon)
       │
       ▼
┌─────────────────────────────────┐
│  HF-MTM Akustik Model (ASR)     │  ← Bu raporun konusu
│  openai/whisper fine-tuned      │
└─────────────┬───────────────────┘
              │ transkripsiyon (verbatim)
              ▼
┌─────────────────────────────────┐
│  OKUMETRİK Motoru               │
│  ├─ miscue.align()              │
│  ├─ phonetics/g2p               │
│  ├─ scoring (WCPM, F1)          │
│  └─ feedback (Türkçe)           │
└─────────────┬───────────────────┘
              │
              ▼
         Rapor (HTML/JSON)
```

---

## 2. Donanım ve Ortam

### 2.1 Donanım

| Bileşen | Değer |
|---|---|
| İşlemci | Intel (Windows 11 Pro) |
| GPU | NVIDIA GeForce RTX 4070 |
| VRAM | 12.9 GB |
| GPU Mimarisi | Ada Lovelace |
| İşletim Sistemi | Windows 11 Pro 10.0.22631 |
| C: Disk (sistem) | ~191 MB boş (kritik sorun — bkz. Bölüm 5) |
| D: Disk (veri) | ~96 GB boş |

### 2.2 Yazılım Ortamı

| Bileşen | Sürüm |
|---|---|
| Python | 3.12.10 |
| Sanal ortam | `.venv` (proje kök dizini) |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| torchaudio | 2.6.0+cu124 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| evaluate | — |
| librosa | 0.11.0 |
| peft | — |
| HF_HOME | `D:\hf_cache` (C: disk doluydu) |

### 2.3 Hassasiyet

RTX 4070 Ada mimarisi **bfloat16 (bf16)** destekler:
```python
use_bf16 = torch.cuda.is_bf16_supported()  # True
use_fp16 = False  # bf16 varken fp16 kullanılmaz
```

---

## 3. Aşama 1 — Offline Pipeline Doğrulama

### 3.1 Amaç

İnternet veya GPU gerektirmeden, WER hesabı ve CTC eğitim döngüsünün uçtan uca çalıştığını kanıtlamak.

**Betik:** `pre_feasibility/selftest_offline.py`

### 3.2 Keşfedilen Sorun — Eksik `wer()` Fonksiyonu

`selftest_offline.py` dosyasında:
```python
from okumetrik.scoring import wer   # ImportError!
```

`okumetrik/scoring.py` dosyasında `wer` fonksiyonu tanımlanmamıştı.

**Uygulanan Çözüm:** Kelime düzeyinde Levenshtein mesafesi hesaplayan DP tabanlı `wer()` fonksiyonu eklendi:

```python
def wer(reference: str, hypothesis: str) -> dict:
    """
    Döndürür: {"S": ikame, "D": silme, "I": ekleme, "N": ref_kelime_sayısı}
    WER = (S + D + I) / N
    """
    r = reference.split()
    h = hypothesis.split()
    N, M = len(r), len(h)
    # ... dinamik programlama (N+1) × (M+1) matris
    S, D, I, _ = dp[N][M]
    return {"S": S, "D": D, "I": I, "N": N}
```

### 3.3 Model — TinyASR (CTC)

```
Mimari: BiLSTM + CTC Head
  Giriş : 13 MFCC özelliği
  LSTM  : 64 gizli birim, çift yönlü
  Head  : Lineer → log_softmax (vocab+1 boyut)
  Kayıp : CTCLoss
```

### 3.4 Sentetik Veri

```
Sözlük  : 10 Türkçe kelime (ali, oku, gel, git, bak, ver, al, dur, sor, sat)
Örnekler: 240 eğitim / 40 test
Format  : Rastgele özellik vektörü + Gaussian gürültü
```

### 3.5 Eğitim Parametreleri (Offline)

| Parametre | Değer |
|---|---|
| Optimizer | Adam |
| Öğrenme oranı | 3e-3 |
| Epoch | 12 |
| Batch size | 8 |

### 3.6 Offline Sonuçlar

| Aşama | WER |
|---|---|
| **Zero-shot (Epoch 0)** | **%100.0** |
| Epoch 3 | %0.0 |
| Epoch 6 | %0.0 |
| Epoch 9 | %0.0 |
| **Fine-tune sonrası (Epoch 12)** | **%0.0** |

**Sonuç:** Pipeline doğrulaması başarılı. WER %100 → %0.0 (sentetik veri üzerinde).

---

## 4. Aşama 2 — Gerçek Whisper Fine-tune (FLEURS)

### 4.1 Model

| Parametre | Değer |
|---|---|
| **Model** | `openai/whisper-small` |
| Parametre sayısı | 244 milyon |
| Tip | Encoder-Decoder Transformer |
| Eğitim tipi | Tam fine-tune (LoRA yok) |
| Dil | Türkçe (`language="turkish"`, `task="transcribe"`) |

### 4.2 Veri Seti

| Parametre | Değer |
|---|---|
| **Veri seti** | `google/fleurs` (`tr_tr` dil kodu) |
| **Açıklama** | Few-shot Learning Evaluation of Universal Representations of Speech |
| **Yayımlayan** | Google Research |
| Toplam veri | ~2526 eğitim / ~743 test örneği |
| **Kullanılan (train)** | **1500 örnek** |
| **Kullanılan (eval)** | **400 örnek** |
| Format | OGG ses (16 kHz) + Türkçe transkript |
| Transkript sütunu | `transcription` |
| Ses çözücü | `librosa` (torchcodec bypass — bkz. Bölüm 5) |

### 4.3 Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|---|---|---|
| `num_train_epochs` | **3** | Toplam epoch sayısı |
| `per_device_train_batch_size` | **16** | GPU başına batch |
| `per_device_eval_batch_size` | **8** | Eval batch |
| `gradient_accumulation_steps` | **1** | Akümülasyon yok |
| `learning_rate` | **1e-4** | Başlangıç öğrenme oranı |
| `warmup_ratio` | **0.1** | İlk %10 warm-up |
| `bf16` | **True** | Ada mimarisi desteği |
| `fp16` | False | bf16 varken kapalı |
| `predict_with_generate` | True | Seq2Seq üretim modu |
| `generation_max_length` | 225 | Max token uzunluğu |
| `eval_strategy` | `"epoch"` | Her epoch'ta değerlendirme |
| `save_strategy` | `"epoch"` | Her epoch'ta kayıt |
| `eval_accumulation_steps` | 1 | VRAM tasarrufu |
| `load_best_model_at_end` | True | En iyi modeli yükle |
| `metric_for_best_model` | `"wer"` | Düşük WER = iyi model |
| `dataloader_num_workers` | 2 | Windows için |
| `logging_steps` | 25 | Log sıklığı |
| `report_to` | `"none"` | WandB/TB kapalı |

### 4.4 Normalleştirme

```python
def normalize_tr(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower().strip()
```

Büyük/küçük harf ve Türkçe İ/I normalizasyonu uygulandı.

### 4.5 Collator

```python
@dataclass
class Collator:
    processor: Any
    def __call__(self, feats):
        # input_features pad
        # labels pad + -100 maskeleme (pad token'ları loss hesabına katılmaz)
```

### 4.6 Eğitim Süresi

| Aşama | Süre |
|---|---|
| Zero-shot WER hesabı (400 örnek) | ~40 dakika |
| Preprocessing (1500 → feature extract) | ~20 dakika |
| Epoch 1 eğitimi | ~1s 10dk |
| Epoch 1 değerlendirmesi | ~2 dakika |
| Epoch 2 eğitimi | ~1s 10dk |
| Epoch 2 değerlendirmesi | ~2 dakika |
| Epoch 3 eğitimi | ~1s 10dk |
| Epoch 3 değerlendirmesi | ~2 dakika |
| **Toplam** | **~6 saat** |

---

## 5. Karşılaşılan Sorunlar ve Çözümler

### 5.1 Eksik `wer()` Fonksiyonu

| | |
|---|---|
| **Hata** | `ImportError: cannot import name 'wer' from 'okumetrik.scoring'` |
| **Neden** | `scoring.py`'de `wer()` tanımlanmamıştı |
| **Çözüm** | DP tabanlı Levenshtein WER fonksiyonu eklendi |
| **Dosya** | `okumetrik/scoring.py` |

### 5.2 CPU-Only PyTorch (CUDA Yok)

| | |
|---|---|
| **Hata** | PyTorch 2.12.1+cpu kuruluydu; RTX 4070 kullanılamıyordu |
| **Belirti** | `torch.cuda.is_available()` → `False` |
| **Çözüm** | CUDA 12.4 destekli PyTorch kuruldu |
| **Komut** | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **Sonuç** | `torch.cuda.is_available()` → `True`, RTX 4070 aktif |

### 5.3 C: Disk Yetersizliği

| | |
|---|---|
| **Hata** | `RuntimeError: Diskte yeterli yer yok` (C: sadece 191 MB boş) |
| **Neden** | HuggingFace modelleri ve veri setleri C:'ye indiriliyordu |
| **Çözüm** | `$env:HF_HOME = "D:\hf_cache"` (D: diskinde 96 GB boş) |
| **Etki** | Tüm model ve veri seti cache'i D:'ye yönlendirildi |

### 5.4 torchcodec / FFmpeg Eksikliği (Windows)

| | |
|---|---|
| **Hata** | `ImportError: To support decoding audio data, please install 'torchcodec'` |
| **Neden** | `datasets >= 3.x` ses çözme için `torchcodec` kullanıyor; Windows'ta FFmpeg DLL eksik |
| **Denenen 1** | `pip install torchcodec` → kuruldu ama FFmpeg DLL yoktu, runtime hatası |
| **Denenen 2** | `pip install datasets==4.8.5` → downgrade, aynı hata |
| **Final çözüm** | `Audio(decode=False)` + `librosa` manuel decode |

```python
# Veri yükleme: ham bayt, çözme yok
ds = ds.cast_column("audio", Audio(decode=False))

# Manuel çözme: librosa → audioread → Windows Media Foundation (OGG desteği)
def decode_audio(audio_item, target_sr=16000):
    import librosa
    raw = audio_item.get("bytes") or audio_item.get("path")
    if isinstance(raw, bytes):
        array, _ = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
    else:
        array, _ = librosa.load(raw, sr=target_sr, mono=True)
    return array.astype(np.float32)
```

### 5.5 transformers 5.x API Değişikliği

| | |
|---|---|
| **Hata** | `TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'` |
| **Neden** | transformers 5.x'te `tokenizer=` argümanı kaldırıldı |
| **Çözüm** | `tokenizer=` → `processing_class=` olarak değiştirildi |

```python
# Eski (transformers <5.x):
trainer = Seq2SeqTrainer(..., tokenizer=processor.feature_extractor)

# Yeni (transformers 5.x):
trainer = Seq2SeqTrainer(..., processing_class=processor.feature_extractor)
```

---

## 6. Fine-tune Sonuçları

### 6.1 WER İlerlemesi

| Epoch | WER | Değişim |
|---|---|---|
| **Zero-shot (Epoch 0)** | **%24.86** | — |
| Epoch 1 | %26.53 | +1.67 (geçici gerileme, normal) |
| Epoch 2 | %22.46 | -4.07 ✅ |
| **Epoch 3 (final)** | **%20.21** | -2.25 ✅ |

**Toplam iyileşme: %24.86 → %20.21 = %4.65 mutlak düşüş (%18.7 göreli)**

> **Not:** Epoch 1'deki geçici gerileme (%24.86 → %26.53) normaldir. Model önce ağırlıklarını bozar, ardından fine-tune verisiyle yeniden organize olur. `load_best_model_at_end=True` sayesinde Epoch 1 modeli değil, Epoch 3 modeli (en iyi WER) kaydedilmiştir.

### 6.2 Eğitim Kayıpları (Training Loss)

| Epoch | Adım | Loss | Grad Norm | LR |
|---|---|---|---|---|
| 1 | ~25 | 1.057 | 10.91 | 8.28e-05 |
| 1 | ~50 | 0.693 | 9.63 | 9.21e-05 |
| 1 | ~75 | 0.562 | 5.99 | 8.22e-05 |
| 1 (eval) | 94 | — | — | — |
| 2 (eval) | 188 | eval_loss: 0.399 | — | — |
| 3 (eval) | 282 | — | — | — |

### 6.3 Değerlendirme Metrikleri

| Epoch | eval_loss | eval_wer | eval_runtime (s) | samples/s |
|---|---|---|---|---|
| 1 | 0.4589 | 26.53 | 164.4 | 2.434 |
| 2 | 0.3990 | 22.46 | 168.1 | 2.379 |
| 3 (final) | — | **20.21** | ~165 | ~2.4 |

### 6.4 Model Kayıt Bilgileri

```
Konum    : D:\hf_cache\whisper-tr-poc\
Model    : openai/whisper-small (fine-tuned)
Format   : SafeTensors shards
Boyut    : ~490 MB
```

### 6.5 Literatür Karşılaştırması

| Sistem | WER (Türkçe) | Veri | Not |
|---|---|---|---|
| whisper-small zero-shot | %24.86 | — | Bu çalışma |
| **whisper-small fine-tuned** | **%20.21** | FLEURS 1500 örn. | **Bu çalışma** |
| Koçak & Ulaş (2024) Whisper+LoRA | %4.3–14.2 | Geniş korpus | Literatür |
| whisper-large-v3 zero-shot | ~%8-12 | — | Beklenti |

---

## 7. Domain-Specific Çocuk ASR Planı

### 7.1 Neden Farklı Bir Yaklaşım Gerekli?

FLEURS yetişkin sesiyle yapılan fine-tune YADOBA hedefine kısmi katkı sağlar. Asıl hedef **6-12 yaş Türkçe çocuk sesli okuması** olduğundan önemli farklar söz konusudur:

| Özellik | Genel ASR (FLEURS) | YADOBA Hedefi |
|---|---|---|
| Konuşmacı | Yetişkin (~20-50 yaş) | 6-12 yaş çocuk |
| Temel frekans (F0) | ~85-255 Hz | ~250-500 Hz |
| Formant frekansları | Düşük | Yüksek (kısa vokal yolu) |
| Okuma hızı | Normal | %8-25 daha yavaş |
| Heceleme | Akıcı | Duraklamalı, tekrarlı |
| ASR çıktı hedefi | Dilbilgisel doğruluk | **Verbatim** (hata dahil) |
| LM beam search | İstenir | **İSTENMEZ** |
| Arka plan gürültüsü | Az | Sınıf/ev ortamı |

### 7.2 Verbatim Transcription

Standart Whisper, dil modeli ile çıktıyı "düzeltebilir":
```
Çocuk der: "karım yidi"  →  Whisper yazar: "karım yedi"  ✗ (miscue kayboldu)
Çocuk der: "karım yidi"  →  YADOBA ister: "karım yidi"   ✓ (miscue korundu)
```

Verbatim mod konfigürasyonu:
```python
model.generation_config.no_repeat_ngram_size = 0   # LM tekrar cezası kapalı
model.config.forced_decoder_ids = None              # Zorunlu token yok

def normalize_verbatim(s: str) -> str:
    """Noktalama kaldır, Türkçe küçük harf — LM düzeltmesi yapma."""
    import re
    s = s.replace("I", "ı").replace("İ", "i").lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()
```

### 7.3 Çocuk Sesi Augmentasyonu

Yetişkin seslerini çocuk sesine yaklaştırmak için üç dönüşüm:

#### 7.3.1 Pitch Shift (Perde Kaydırma)

```python
def pitch_shift_child(audio, sr=16000, semitones_range=(3, 7)):
    import librosa
    n_steps = random.uniform(*semitones_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
```

| Parametre | Değer | Gerekçe |
|---|---|---|
| Kaydırma miktarı | +3 ila +7 yarı ton | Yetişkin F0 ~120 Hz → Çocuk F0 ~250 Hz (+12 yarı ton ≈ ×2) |
| Uygulama | rastgele seçim | Çocuklar arasındaki doğal varyasyon |

#### 7.3.2 Tempo Yavaşlatma

```python
def tempo_slow(audio, sr=16000, rate_range=(0.75, 0.92)):
    import librosa
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)
```

| Parametre | Değer | Gerekçe |
|---|---|---|
| Hız oranı | ×0.75 – ×0.92 | Çocuklar yetişkinlerden %8-25 yavaş okur |
| Yöntem | time_stretch (pitch korunur) | Pitch değişmeden sadece hız azalır |

#### 7.3.3 Oda Gürültüsü

```python
def add_room_noise(audio, snr_db_range=(15, 30)):
    snr_db = random.uniform(*snr_db_range)
    noise_power = np.mean(audio**2) / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)
```

| Parametre | Değer | Gerekçe |
|---|---|---|
| SNR | 15-30 dB | Sınıf/ev ortamı gürültüsü seviyesi |
| Gürültü tipi | Gaussian (beyaz) | Genel arka plan simülasyonu |

### 7.4 Sentetik Okul Metni Veri Seti

`synth_school_data.py` ile üretilen veri:

- **60+ cümle:** İlkokul 1-4. sınıf Türkçe okuma metinlerinden
- **TTS motoru:** gTTS (Google Text-to-Speech) — Türkçe
- **Augmentasyon:** pitch + tempo + gürültü (isteğe bağlı)
- **Format:** numpy array + transkript → HuggingFace Dataset

Örnek cümleler (sınıf seviyesine göre):
```
1. sınıf : "Ali oku.", "Top kırmızıdır.", "Kuş uçuyor."
2. sınıf : "Bugün hava çok güzel.", "Kelebek çiçeğe kondu."
3. sınıf : "Öğretmenimiz bize hikâye anlattı."
4. sınıf : "Tarihî eserleri korumak hepimizin görevidir."
Akıcılık : "Ali dün sabah erkenden kalktı ve okula gitmek için hazırlandı."
```

### 7.5 Veri Kaynakları Hiyerarşisi

| Katman | Kaynak | Durum | Miktar |
|---|---|---|---|
| 1 | FLEURS tr_tr + augment | ✅ Hazır | 1500 örnek |
| 2 | Common Voice 17 tr + augment | ✅ Kod hazır | ~1000 örnek |
| 3 | Sentetik okul metni (gTTS) | ✅ Kod hazır | 60+ cümle |
| 4 | Gerçek çocuk korpusu (etik onaylı) | ⏳ İP3 fazı | Belirsiz |

### 7.6 Domain Fine-tune Modeli

**Betik:** `pre_feasibility/finetune_whisper_child_tr.py`

| Parametre | Değer |
|---|---|
| **Temel model** | `openai/whisper-large-v3` |
| **Parametre sayısı** | 1.5 milyar |
| **Eğitim tipi** | LoRA (VRAM tasarrufu) |
| `lora_r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | q_proj, v_proj, k_proj, out_proj |
| `batch_size` | 4 |
| `grad_accum` | 4 (efektif batch=16) |
| `gradient_checkpointing` | True |
| `epochs` | 5 |
| `lr` | 5e-5 |
| Tahmini VRAM | ~10-11 GB (RTX 4070 sınırında ✅) |

---

## 8. WER %5 Hedefine Yol Haritası

### 8.1 Mevcut Durum ve Hedefler

```
Tamamlanan:
  ✅ Aşama 0: Offline CTC pipeline  → WER %100 → %0.0 (sentetik)
  ✅ Aşama 1: whisper-small + FLEURS → WER %24.86 → %20.21 (gerçek)

Planlanan:
  ⏳ Aşama 2: whisper-large-v3 + LoRA + FLEURS       → beklenti %12-18
  ⏳ Aşama 3: + pitch/tempo augmentasyon              → beklenti %10-15
  ⏳ Aşama 4: + sentetik okul metni                   → beklenti %8-12
  ⏳ Aşama 5: + Common Voice 17 tr                    → beklenti %7-10
  ⏳ Aşama 6: + gerçek çocuk korpusu (İP3/İP4)        → hedef %5-8
```

### 8.2 WER Azaltma Faktörleri

| Faktör | Tahmini WER Kazancı |
|---|---|
| Modeli small → large-v3 | -5 ila -8 puan |
| LoRA ile domain fine-tune | -2 ila -4 puan |
| Çocuk augmentasyonu | -1 ila -3 puan |
| Sentetik + Common Voice verisi | -1 ila -2 puan |
| Gerçek çocuk korpusu | -3 ila -6 puan |
| **Toplam (kümülatif)** | **~-12 ila -20 puan** |

### 8.3 Sonraki Çalıştırılacak Komut

```powershell
# Temel: large-v3 + LoRA + augmentasyon + sentetik
$env:HF_HOME = "D:\hf_cache"
$py = "C:\Users\Murat\Desktop\yadoba-okumetrik\.venv\Scripts\python.exe"
& $py pre_feasibility/finetune_whisper_child_tr.py `
    --epochs 5 --augment --synth --synth_n 60

# Tam: Common Voice da dahil
& $py pre_feasibility/finetune_whisper_child_tr.py `
    --epochs 5 --augment --synth --synth_n 60 --common_voice
```

---

## 9. Oluşturulan / Değiştirilen Dosyalar

| Dosya | Değişiklik / İçerik |
|---|---|
| `okumetrik/scoring.py` | `wer(reference, hypothesis) -> dict` eklendi |
| `pre_feasibility/finetune_whisper_tr.py` | Whisper fine-tune script (torchcodec bypass, bf16, transformers 5.x uyumu) |
| `pre_feasibility/augment_child.py` | Çocuk sesi augmentasyon araçları (pitch, tempo, gürültü) |
| `pre_feasibility/synth_school_data.py` | gTTS ile okul metni sentetik veri üretici |
| `pre_feasibility/finetune_whisper_child_tr.py` | Domain-specific whisper-large-v3 + LoRA + verbatim fine-tune |
| `TEKNIK_RAPOR.md` | Bu dosya |

---

## 10. Referanslar

1. Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). OpenAI.
2. Koçak, M. & Ulaş, A. H. (2024). *Whisper ile Türkçe ASR: LoRA Fine-tuning Değerlendirmesi.* — WER %4.3-14.2 referansı.
3. Google Research (2022). *FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech.*
4. Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
5. HuggingFace Transformers — `Seq2SeqTrainer`, `WhisperForConditionalGeneration`.
6. TÜBİTAK 1501 Başvurusu — YADOBA 2.0, AGY100 teknik şartnamesi.

---

*Bu rapor 2026-06-19 tarihinde YADOBA 2.0 / OKUMETRİK projesi kapsamında hazırlanmıştır.*
