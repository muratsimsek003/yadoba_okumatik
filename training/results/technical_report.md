# YADOBA — Teknik Rapor: Konuşma Tanıma Model Karşılaştırması

**Tarih:** 2026-07-01  
**Değerlendirme:** FLEURS Türkçe test seti, 200 örnek  
**Görev:** Otomatik konuşma tanıma (ASR), Türkçe  

---

## 1. Yöntem

### 1.1 Değerlendirme Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | google/fleurs, tr_tr, split="test" |
| Örnek sayısı | 200 (tam test: ~674) |
| Ses formatı | 16 kHz, mono, WAV |
| Metin normalizasyonu | küçük harf, noktalama kaldırma, tekli boşluk |

### 1.2 Metrikler

- **WER (Word Error Rate):** `(S + D + I) / N × 100` — düşük iyi
- **CER (Character Error Rate):** karakter düzeyinde WER — düşük iyi
- **Word-F1:** kelime örtüşmesi üzerinden F1 skoru — yüksek iyi

### 1.3 Test Edilen Modeller

| Anahtar | Model | Backend | Açıklama |
|---------|-------|---------|----------|
| `whisper-small` | openai/whisper-small | HuggingFace | Mevcut VPS modeli, temel çizgi |
| `whisper-medium-zero` | openai/whisper-medium | HuggingFace | İnce ayar öncesi, sıfır-atış |
| `whisper-medium-ft` | D:\yadoba-training\medium-tr-yadoba-ct2 | CTranslate2 int8 | FLEURS üzerinde LoRA ince ayar |
| `whisper-large-v3-ft` | D:\ct2_whisper_yadoba | CTranslate2 int8 | Adım 196 checkpoint, LoRA |

---

## 2. Sonuçlar

### 2.1 Karşılaştırma Tablosu

| Model | WER% ↓ | CER% ↓ | Precision% ↑ | Recall% ↑ | Word-F1% ↑ | Süre (s) |
|-------|--------|--------|-------------|----------|-----------|---------|
| Whisper Small (VPS baseline) | **16.58** | **4.01** | 84.59 | 84.56 | 84.52 | 179 |
| Whisper Medium (zero-shot) | **10.65** | **2.73** | 90.48 | 90.38 | 90.37 | 536 |
| Whisper Medium (YADOBA LoRA) | 407.50 | 160.16 | 6.50 | 2.97 | 3.98 | 330 |
| Whisper Large-v3 (LoRA chk-196) | 125.51 | 82.05 | 4.39 | 4.69 | 4.52 | 203 |

> **Not:** WER > 100%, modelin referans metninden çok daha uzun çıktı ürettiğini gösterir (halüsinasyon/tekrar döngüsü).

### 2.2 Temel Bulgular

1. **En iyi model:** Whisper Medium zero-shot — WER %10.65, Word-F1 %90.37
2. **Mevcut VPS modeli (Whisper Small):** WER %16.58, iyi bir temel çizgi
3. **İnce ayarlı modeller:** Her ikisi de başarısız — modeller tutarlı konuşma metni yerine halüsinasyon üretiyor

---

## 3. İnce Ayar Analizi: Neden Başarısız?

### 3.1 Whisper Medium LoRA (WER %407.5)

**Eğitim parametreleri:**
- Baz model: openai/whisper-medium (~307M)
- LoRA: r=64, alpha=128, hedefler: q\_proj, v\_proj, out\_proj, fc1, fc2
- Dataset: FLEURS tr\_tr (2526 train, 338 val)
- Adım: 3000, batch: 8 × grad\_acc 4 = etkin batch 32
- **Learning rate: 1e-3** (cosine)
- fp16

**Gözlemlenen belirtiler:**

| Gösterge | Değer | Yorumlama |
|----------|-------|-----------|
| Eğitim kaybı (son) | 19.25 | Rassal tahmin eşiği (~10.8) **üzerinde** |
| Eval kaybı (son) | 6.236 | Yüksek, yakınsamama işareti |
| WER | %407.5 | Model halüsinasyon yapıyor |
| CER | %160.2 | Karakter düzeyinde de anlamsız çıktı |

**Muhtemel nedenler:**

1. **Öğrenme oranı çok yüksek (1e-3):** Whisper LoRA için tipik değer 1e-4 – 5e-5'tir. 1e-3, özellikle büyük LoRA rank (r=64) ile birleşince gradyan patlamasına yol açmış olabilir.

2. **Eğitim kaybı rassal eşiğin üzerinde:** 51000 kelimelik Whisper tokenizer'ında rassal tahmin kaybı ln(51000) ≈ 10.8'dir. Eğitim kaybı 19.25, modelin rastgele tahmin bile yapamadığını gösterir — model muhtemelen eğitim sırasında sapmiştir.

3. **PEFT + Whisper forward uyumsuzluğu:** `PeftModelForSeq2SeqLM.forward()` Whisper'ın `input_features` yerine `input_ids` beklediğinden monkey-patch uygulandı. Bu patch eğitimde label hesaplamasını etkilemiş olabilir.

4. **Decoder start token kayması:** `DataCollatorSpeechSeq2SeqWithPadding` decoder\_start\_token'ı etiketlerden siliyor. Eğer bu işlem PEFT modeline yanlış aktarılırsa label hizalaması bozulur.

### 3.2 Whisper Large-v3 LoRA (WER %125.5)

Bu model henüz checkpoint 196'da (çok erken aşama). Düşük adım sayısıyla model tam olarak yakınsamadan test edilmiştir. Bununla birlikte WER %125.5 hâlâ kabul edilemez düzeyde.

---

## 4. Mevcut Sistem Değerlendirmesi

YADOBA şu an VPS'te **Whisper Small** kullanmaktadır:

- WER %16.58, Word-F1 %84.52
- Çocuk okuma analizinde kabul edilebilir performans
- Düşük bellek/GPU gereksinimiyle hızlı çalışıyor

**Whisper Medium zero-shot** ise %36 daha iyi WER sunuyor (10.65 vs 16.58) — ekstra fine-tune gerekmeden doğrudan upgrade yapılabilir.

---

## 5. İyileştirme Önerileri

### 5.1 Kısa Vadeli: Whisper Medium Zero-Shot'a Geçiş

VPS'e Whisper Medium modelini kurmak basit bir iyileştirme sağlar:

| | Whisper Small | Whisper Medium (zero-shot) |
|-|--------------|---------------------------|
| WER | %16.58 | %10.65 |
| Word-F1 | %84.52 | %90.37 |
| Model boyutu | ~244 MB | ~769 MB |
| Bellk kullanımı | ~1 GB | ~2-3 GB |

### 5.2 Orta Vadeli: Doğru Hiperparametrelerle Yeniden Fine-Tune

Eğer LoRA fine-tune tekrarlanacaksa:

```python
# Önerilen düzeltmeler
learning_rate    = 1e-4     # 1e-3 yerine (10× düşük)
warmup_steps     = 500      # daha uzun ısınma
max_grad_norm    = 0.5      # gradyan kırpma ekle
lora_r           = 32       # 64 yerine (daha stabil)
lora_alpha       = 64       # rank ile orantılı
```

Ayrıca DataCollator'da decoder start token işleme ve PEFT forward patch'in eğitim modunda label hesaplamayı bozmadığı doğrulanmalıdır.

### 5.3 Uzun Vadeli: Türkçe Çocuk Konuşması Verisi

FLEURS yetişkin konuşması içeriyor. YADOBA için ideal senaryo, gerçek çocuk okuma kayıtları üzerinde fine-tune yapmaktır. Bu, hem WER'i düşürecek hem de yanlış telaffuz/hece hataları gibi pedagojik açıdan değerli bilgileri modele öğretecektir.

---

## 6. Sonuç

| Soru | Yanıt |
|------|-------|
| Fine-tune başarılı mı? | Hayır — her iki model halüsinasyon üretiyor |
| En iyi mevcut seçenek? | Whisper Medium zero-shot (WER %10.65) |
| Mevcut VPS yeterli mi? | Evet, Whisper Small iyi bir temel çizgi |
| Fine-tune tekrarlanmalı mı? | Evet, düşük LR + gradyan kırpma ile |

**Birincil öneri:** VPS'i Whisper Medium zero-shot'a yükselt. Bu, sıfır risk ve %36 WER iyileştirmesiyle en kolay kazanımdır.

---

*Değerlendirme kodu: `training/evaluate_wer.py`*  
*Ham sonuçlar: `training/results/wer_comparison.json`*  
*Eğitim scripti: `training/finetune_medium.py`*
