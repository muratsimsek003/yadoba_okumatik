# Teknik Döküman — YADOBA ASR Geliştirme Pipeline'ı

**Tarih:** 2026-06-19  
**Kapsam:** Offline WER doğrulama + Gerçek Whisper fine-tune + Domain-specific çocuk ASR planı  
**Proje:** YADOBA / Okumetrik — Çocuk Sesli Okuma Değerlendirme Sistemi

---

## 1. Amaç

Bu döküman, `selftest_offline.py` betiğinin başarıyla çalıştırılması sürecini, karşılaşılan eksikliği ve uygulanan çözümü açıklamaktadır.

Betiğin hedefi: internet veya GPU gerektirmeden, bir CTC (Connectionist Temporal Classification) eğitim döngüsünün ve WER (Word Error Rate) hesabının uçtan uca çalıştığını kanıtlamak. Çıktı olarak **zero-shot WER** (eğitim öncesi) ile **fine-tune sonrası WER** karşılaştırması üretilir.

---

## 2. Ortam

| Bileşen | Değer |
|---|---|
| İşletim sistemi | Windows 11 Pro (10.0.22631) |
| Python | 3.12.10 (`.venv` sanal ortamı) |
| PyTorch | 2.12.1+cpu (CPU build, CUDA yok) |
| GPU | — (offline demo; GPU gerekmez) |
| İnternet | — (offline demo; HuggingFace gerekmez) |

---

## 3. Karşılaşılan Sorun

`selftest_offline.py` dosyasının 18. satırında şu import ifadesi bulunmaktaydı:

```python
from okumetrik.scoring import wer
```

Ancak `okumetrik/scoring.py` dosyasında `wer` adında bir fonksiyon **tanımlanmamıştı**. Dosya yalnızca okuma akıcılığı metriklerini (`summarize`, `prf`, `evaluate_miscues`) içeriyordu.

Betik çalıştırıldığında alınan hata:

```
ImportError: cannot import name 'wer' from 'okumetrik.scoring'
```

---

## 4. Uygulanan Çözüm

`okumetrik/scoring.py` dosyasına standart **edit-distance tabanlı WER** fonksiyonu eklendi.

### 4.1 Fonksiyon İmzası

```python
def wer(reference: str, hypothesis: str) -> dict:
```

**Parametreler:**
- `reference`: Referans (doğru) metin
- `hypothesis`: Model tarafından tahmin edilen metin

**Döndürür:**
```python
{"S": <ikame sayısı>, "D": <silme sayısı>, "I": <ekleme sayısı>, "N": <ref kelime sayısı>}
```

### 4.2 Algoritma

Levenshtein mesafesinin kelime düzeyindeki uyarlaması; standart **dinamik programlama (DP)** ile çözülür.

```
dp[i][j] = referansın ilk i kelimesini, hipotezin ilk j kelimesiyle
           hizalamanın minimum maliyeti (S, D, I bileşenleriyle birlikte)
```

Geçiş kuralları:

| Durum | İşlem | Maliyet |
|---|---|---|
| `r[i] == h[j]` | Eşleşme | 0 |
| `r[i] != h[j]` | İkame (S) | +1 |
| `h[j]` eksik | Silme (D) | +1 |
| `r[i]` eksik | Ekleme (I) | +1 |

### 4.3 Eklenen Kod (`okumetrik/scoring.py`)

```python
def wer(reference: str, hypothesis: str) -> dict:
    r = reference.split()
    h = hypothesis.split()
    N, M = len(r), len(h)
    INF = float("inf")
    dp = [[(INF, 0, 0, 0)] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for j in range(1, M + 1):
        dp[0][j] = (0, 0, 0, j)        # j ekleme
    for i in range(1, N + 1):
        dp[i][0] = (0, i, 0, 0)        # i silme
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if r[i - 1] == h[j - 1]:
                s, d, ins, _ = dp[i - 1][j - 1]
                dp[i][j] = (s, d, ins, s + d + ins)
            else:
                sub = dp[i - 1][j - 1]; cost_sub = sub[0] + sub[1] + sub[2] + 1
                dlt = dp[i - 1][j];     cost_del = dlt[0] + dlt[1] + dlt[2] + 1
                ins = dp[i][j - 1];     cost_ins = ins[0] + ins[1] + ins[2] + 1
                if cost_sub <= cost_del and cost_sub <= cost_ins:
                    dp[i][j] = (sub[0] + 1, sub[1], sub[2], cost_sub)
                elif cost_del <= cost_ins:
                    dp[i][j] = (dlt[0], dlt[1] + 1, dlt[2], cost_del)
                else:
                    dp[i][j] = (ins[0], ins[1], ins[2] + 1, cost_ins)
    S, D, I, _ = dp[N][M]
    return {"S": S, "D": D, "I": I, "N": N}
```

---

## 5. Kurulum

PyTorch önceden yüklü değildi. Sanal ortama CPU build kuruldu:

```powershell
cd "c:\Users\Murat\Desktop\yadoba-okumetrik"
.\.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu
```

> **Not:** `numpy` kurulu değil; torch çalışırken bir `UserWarning` verir ama pipeline'ı etkilemez. `pip install numpy` ile giderilebilir.

---

## 6. Çalıştırma

```powershell
cd "c:\Users\Murat\Desktop\yadoba-okumetrik"
.\.venv\Scripts\python.exe pre_feasibility/selftest_offline.py
```

---

## 7. Sonuçlar

| Aşama | WER |
|---|---|
| **Zero-shot (eğitim öncesi)** | **%100.0** |
| Epoch 3 | %0.0 |
| Epoch 6 | %0.0 |
| Epoch 9 | %0.0 |
| **Fine-tune sonrası (Epoch 12)** | **%0.0** |

**Örnek tahminler (referans → model çıktısı):**

```
'oku at topu koştu kedi'   →  'oku at topu koştu kedi'   ✓
'kedi ali at'              →  'kedi ali at'               ✓
'oku at topu gel kitap'    →  'oku at topu gel kitap'     ✓
'kitap git at'             →  'kitap git at'              ✓
```

**Sonuç:** Pipeline doğrulama başarılı. CTC eğitim döngüsü ve WER hesabı uçtan uca çalışmaktadır. Eğitim WER'i **%100 → %0.0** düşürdü.

---

## 8. Pipeline Mimarisi

```
selftest_offline.py
│
├── Sentetik Veri Üretimi
│   ├── LEXICON: 10 Türkçe kelime
│   ├── 240 eğitim / 40 test örneği
│   └── Her kelime: rastgele özellik vektörü + gürültü
│
├── Model: TinyASR (BiLSTM + CTC)
│   ├── LSTM: 64 gizli birim, çift yönlü
│   └── Head: lineer katman → log_softmax
│
├── Eğitim
│   ├── Kayıp: CTCLoss
│   ├── Optimizer: Adam (lr=3e-3)
│   └── 12 epoch
│
├── Değerlendirme
│   ├── Greedy CTC decode
│   └── WER hesabı: okumetrik.scoring.wer()  ← eklenen fonksiyon
│
└── Çıktı
    ├── Zero-shot WER (eğitim öncesi)
    └── Fine-tune sonrası WER
```

---

## 9. Sınırlılıklar

- Bu **sentetik bir doğrulamadır**; gerçek Türkçe ASR performansını ölçmez.
- Gerçek Türkçe WER için `pre_feasibility/finetune_whisper_tr.py` kullanılmalıdır (GPU + HuggingFace erişimi gerektirir).
- Literatürde Whisper + LoRA ile Türkçe için beklenen değerler: **WER ~%4.3–14.2** (Koçak & Ulaş, 2024).

---

## 10. Değiştirilen Dosyalar

| Dosya | Değişiklik |
|---|---|
| `okumetrik/scoring.py` | `wer(reference, hypothesis) -> dict` fonksiyonu eklendi |

---

## 11. Gerçek Whisper Fine-tune — Sonuçlar (FLEURS tr_tr)

**Model:** `openai/whisper-small` (244M parametre, tam fine-tune)  
**Veri seti:** `google/fleurs` Türkçe — 1500 train / 400 eval  
**Donanım:** RTX 4070 12 GB | bf16 | 3 epoch | batch_size=16

| Aşama | WER |
|---|---|
| **Zero-shot (fine-tune öncesi)** | **%24.86** |
| Epoch 1 | %26.53 |
| Epoch 2 | ⏳ devam ediyor |
| Epoch 3 (final) | ⏳ devam ediyor |

> Epoch 1'de hafif gerileme normaldir; model fine-tune veri dağılımına uyum sağlamaktadır.
> `load_best_model_at_end=True` ile en iyi checkpoint seçilecektir.

**Çözülen Teknik Sorunlar:**

| Sorun | Çözüm |
|---|---|
| `torchcodec` Windows'ta FFmpeg DLL eksikliği | `Audio(decode=False)` + `librosa` manuel decode |
| `Seq2SeqTrainer(tokenizer=...)` transformers 5.x hatası | `processing_class=` argümanına geçiş |
| C: disk dolu (191 MB) | `$env:HF_HOME = "D:\hf_cache"` |
| CPU-only PyTorch | `torch 2.6.0+cu124` + `torchaudio 2.6.0+cu124` |

---

## 12. Domain-Specific Çocuk ASR Planı (YADOBA)

### 12.1 Neden Domain-Specific Gerekli?

FLEURS ve genel Türkçe ASR modelleri yetişkin sesine göre eğitilmiştir. YADOBA'nın hedef kitlesi **6-12 yaş Türkçe çocuk** sesli okumasıdır:

| Özellik | Genel ASR | YADOBA |
|---|---|---|
| Konuşmacı | Yetişkin | 6-12 yaş çocuk |
| Temel frekans | ~85-255 Hz | ~250-500 Hz |
| Okuma hızı | Normal | %8-25 daha yavaş |
| ASR hedefi | Dilbilgisel doğruluk | **Verbatim** — hata dahil |
| LM düzeltmesi | İstenir | **İSTENMEZ** |

### 12.2 Verbatim Transcription

Standart Whisper "karım yidi" → "karım yedi" düzeltebilir.
YADOBA'da çocuk "karım yidi" dediyse **"karım yidi"** yazılmalıdır; bu bir miscue'dur.

Uygulanan yapılandırma:
```python
model.generation_config.no_repeat_ngram_size = 0   # LM cezasını kapat
model.config.forced_decoder_ids = None              # token zorlamasını kaldır
# normalize_verbatim(): noktalama kaldır, Türkçe küçük harf — LM düzeltmesi yok
```

### 12.3 Veri Stratejisi

```
Katman 1: FLEURS tr_tr (yetişkin) + pitch/tempo augment → çocuk simülasyonu
Katman 2: Common Voice 17 tr    + pitch/tempo augment → çeşitli aksanlar
Katman 3: Sentetik okul metni   (gTTS + çocuk augment) → domain kelime hazinesi
Katman 4: [İP3] Gerçek çocuk verisi (etik onaylı korpus) → nihai kalibrasyon
```

### 12.4 Augmentasyon Parametreleri

| Dönüşüm | Aralık | Amaç |
|---|---|---|
| Pitch shift | +3 ila +7 yarı ton | Çocuk temel frekansı simülasyonu |
| Tempo yavaşlat | ×0.75 – ×0.92 | Çocuk okuma hızı |
| Oda gürültüsü | SNR 15-30 dB | Sınıf/ev ortamı |

### 12.5 Yeni Dosyalar

| Dosya | Amaç |
|---|---|
| `pre_feasibility/augment_child.py` | Pitch/tempo/gürültü augmentasyon araçları |
| `pre_feasibility/synth_school_data.py` | gTTS ile okul metni sentetik veri üretimi |
| `pre_feasibility/finetune_whisper_child_tr.py` | whisper-large-v3 + LoRA + verbatim fine-tune |

### 12.6 Geliştirme Yol Haritası

```
✅ Aşama 0: Offline pipeline doğrulama (selftest_offline.py)
✅ Aşama 1: whisper-small + FLEURS baseline fine-tune (devam ediyor)
⏳ Aşama 2: whisper-large-v3 + LoRA + FLEURS (güçlü genel baseline)
⏳ Aşama 3: + pitch/tempo augmentasyon (çocuk simülasyonu)
⏳ Aşama 4: + sentetik okul metni (gTTS, 60+ cümle)
⏳ Aşama 5: + Common Voice 17 tr (geniş aksan çeşitliliği)
⏳ Aşama 6: [İP3] + gerçek çocuk korpusu (etik onaylı)
```

### 12.7 WER Hedefleri

| Aşama | Beklenen WER | Not |
|---|---|---|
| Aşama 1 (şu an) | %18-24 | whisper-small, genel Türkçe |
| Aşama 2 | %12-18 | whisper-large-v3 + LoRA |
| Aşama 3-4 | %10-15 | çocuk augment + sentetik |
| Aşama 5 | %8-12 | geniş veri |
| Aşama 6 | **%5-8** | gerçek çocuk verisi, domain-specific |
