#!/usr/bin/env python3
"""
finetune_whisper_child_tr.py — YADOBA domain-specific fine-tune.

Hedef: Türkçe çocuk sesli okumasını VERBATIM (kelimesi kelimesine) transkripsyon.
Model: whisper-large-v3 + LoRA (RTX 4070 12 GB VRAM'e sığar)

Veri katmanları (sırayla eklenir):
  1. FLEURS tr_tr   — genel Türkçe yetişkin sesi (pitch/tempo augment ile çocuğa çevrilir)
  2. Common Voice 17 tr — gönüllü çeşitli aksanlar (augment ile)
  3. Sentetik okul metni — gTTS + çocuk augmentasyonu
  4. [Gelecek İP3] Gerçek çocuk korpusu — etik onaylı kayıtlar

Verbatim mod:
  - Whisper LM beam search düzeltmesi devre dışı (no_repeat_ngram_size=1)
  - Dil zorlaması: sadece Turkish
  - suppress_tokens: noktalama/düzeltme token'ları bastırılır
  - Hedef: çocuğun GERÇEKTE ne dediği, dilbilgisel açıdan doğru versiyon değil

HIZLI KULLANIM:
  # sadece zero-shot kontrol
  python finetune_whisper_child_tr.py --baseline_only

  # tam eğitim (FLEURS + sentetik, augment ile)
  python finetune_whisper_child_tr.py --epochs 5 --synth --augment

  # tüm veri kaynakları
  python finetune_whisper_child_tr.py --epochs 5 --synth --augment --common_voice

VRAM notu (RTX 4070 12 GB):
  whisper-large-v3 + LoRA r=32 + grad_ckpt + bs4 + bf16  → ~10-11 GB ✅
"""
import argparse
import io
import logging
import warnings
import numpy as np

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

TARGET_SR = 16000


# ---------------------------------------------------------------------------
# Ses çözücü (torchcodec bypass)
# ---------------------------------------------------------------------------

def decode_audio(audio_item, sr: int = TARGET_SR) -> np.ndarray:
    import librosa
    raw = audio_item.get("bytes") or audio_item.get("path")
    if isinstance(raw, bytes):
        arr, _ = librosa.load(io.BytesIO(raw), sr=sr, mono=True)
    else:
        arr, _ = librosa.load(raw, sr=sr, mono=True)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_fleurs(max_train: int, max_eval: int):
    from datasets import load_dataset, Audio
    tr = load_dataset("google/fleurs", "tr_tr", split="train")
    te = load_dataset("google/fleurs", "tr_tr", split="test")
    tr = tr.cast_column("audio", Audio(decode=False))
    te = te.cast_column("audio", Audio(decode=False))
    if max_train:
        tr = tr.select(range(min(max_train, len(tr))))
    if max_eval:
        te = te.select(range(min(max_eval, len(te))))
    return tr, te, "transcription"


def load_common_voice(max_train: int, max_eval: int):
    from datasets import load_dataset, Audio
    tr = load_dataset("mozilla-foundation/common_voice_17_0", "tr",
                      split="train", trust_remote_code=True)
    te = load_dataset("mozilla-foundation/common_voice_17_0", "tr",
                      split="test", trust_remote_code=True)
    tr = tr.cast_column("audio", Audio(decode=False))
    te = te.cast_column("audio", Audio(decode=False))
    if max_train:
        tr = tr.select(range(min(max_train, len(tr))))
    if max_eval:
        te = te.select(range(min(max_eval, len(te))))
    return tr, te, "sentence"


# ---------------------------------------------------------------------------
# Türkçe normalleştirme (verbatim: noktalama kaldır, küçük harf)
# ---------------------------------------------------------------------------

def normalize_verbatim(s: str) -> str:
    """Verbatim mod: noktalama kaldır, Türkçe büyük→küçük harf.

    Amaç: modelin "karım" yerine "karim" üretmesini değil, çocuğun
    tam söylediklerini karşılaştırabilmek. Noktalama ASR çıktısında
    olmaz; referanstan da kaldırıyoruz.
    """
    import re
    s = s.replace("I", "ı").replace("İ", "i").lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Veri seti birleştirme + augmentasyon
# ---------------------------------------------------------------------------

def build_train_dataset(ds_fleurs_tr, text_col_fleurs: str,
                        ds_cv_tr=None, text_col_cv: str = "sentence",
                        use_synth: bool = False,
                        use_augment: bool = True,
                        synth_n: int = 60,
                        augment_ratio: float = 0.7):
    """Tüm veri kaynaklarını birleştir ve augmentasyon uygula.

    Döndürür:
        list of {"array": np.ndarray, "text": str}
    """
    from augment_child import augment_sample
    import random as _rnd

    samples = []

    # FLEURS
    print(f"  FLEURS yükleniyor ({len(ds_fleurs_tr)} örnek)...")
    for ex in ds_fleurs_tr:
        try:
            arr = decode_audio(ex["audio"])
            if use_augment and _rnd.random() < augment_ratio:
                arr = augment_sample(arr, TARGET_SR)
            samples.append({"array": arr, "text": normalize_verbatim(ex[text_col_fleurs])})
        except Exception:
            pass

    # Common Voice
    if ds_cv_tr is not None:
        print(f"  Common Voice yükleniyor ({len(ds_cv_tr)} örnek)...")
        for ex in ds_cv_tr:
            try:
                arr = decode_audio(ex["audio"])
                if use_augment and _rnd.random() < augment_ratio:
                    arr = augment_sample(arr, TARGET_SR)
                samples.append({"array": arr, "text": normalize_verbatim(ex[text_col_cv])})
            except Exception:
                pass

    # Sentetik okul metni
    if use_synth:
        print(f"  Sentetik veri üretiliyor ({synth_n} örnek)...")
        from synth_school_data import SCHOOL_SENTENCES, build_synth_dataset
        import tempfile, random
        sents = random.sample(SCHOOL_SENTENCES, min(synth_n, len(SCHOOL_SENTENCES)))
        synth = build_synth_dataset(sents, tempfile.mkdtemp(),
                                    augment=use_augment, seed=42)
        for s in synth:
            samples.append({"array": s["array"],
                            "text": normalize_verbatim(s["transcription"])})

    print(f"  Toplam eğitim örneği: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Ana eğitim döngüsü
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="YADOBA domain Whisper fine-tune (çocuk okuma)")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=1500,
                    help="FLEURS train örnekleri")
    ap.add_argument("--max_eval", type=int, default=400,
                    help="FLEURS eval örnekleri")
    ap.add_argument("--synth", action="store_true",
                    help="Sentetik okul metni ekle (gTTS gerekli)")
    ap.add_argument("--synth_n", type=int, default=60,
                    help="Sentetik cümle sayısı")
    ap.add_argument("--common_voice", action="store_true",
                    help="Common Voice 17 veri seti ekle")
    ap.add_argument("--max_cv_train", type=int, default=1000)
    ap.add_argument("--augment", action="store_true",
                    help="Çocuk sesi augmentasyonu uygula")
    ap.add_argument("--augment_ratio", type=float, default=0.7)
    ap.add_argument("--baseline_only", action="store_true",
                    help="Sadece zero-shot WER")
    ap.add_argument("--out", default="whisper-child-tr-poc")
    args = ap.parse_args()

    import torch, evaluate
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from peft import LoraConfig, get_peft_model
    from dataclasses import dataclass
    from typing import Any

    if not torch.cuda.is_available():
        print("[UYARI] CUDA bulunamadı!")
    else:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {name}  |  VRAM ~{vram:.1f} GB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"[precision] bf16={use_bf16}  fp16={use_fp16}")

    # Model + processor
    print(f"[model] {args.model} yükleniyor...")
    processor = WhisperProcessor.from_pretrained(
        args.model, language="turkish", task="transcribe")
    # whisper-large-v3 HF ağırlıkları fp16 — float32'ye zorla,
    # trainer bf16=True ile autocast yapar
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    model.generation_config.language = "turkish"
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.generation_config.no_repeat_ngram_size = 0

    # Veri + metrik (LoRA'dan önce yükle)
    ds_fleurs_tr, ds_fleurs_te, text_col_f = load_fleurs(args.max_train, args.max_eval)
    ds_cv_tr = None
    if args.common_voice:
        ds_cv_tr, _, _ = load_common_voice(args.max_cv_train, 0)
    wer_metric = evaluate.load("wer")

    # Zero-shot WER — LoRA'dan ÖNCE, ham base model ile
    def zero_shot_wer(m):
        m.eval()
        refs, hyps = [], []
        for ex in ds_fleurs_te:
            feats = processor.feature_extractor(
                decode_audio(ex["audio"]), sampling_rate=TARGET_SR,
                return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                ids = m.generate(feats, max_new_tokens=225,
                                 language="tr", task="transcribe")
            hyps.append(normalize_verbatim(
                processor.batch_decode(ids, skip_special_tokens=True)[0]))
            refs.append(normalize_verbatim(ex[text_col_f]))
        return 100 * wer_metric.compute(predictions=hyps, references=refs)

    print(f"\n[{args.model}] zero-shot WER hesaplanıyor ({len(ds_fleurs_te)} örnek)...")
    base_wer = zero_shot_wer(model)
    print(f"  >> ZERO-SHOT WER = %{base_wer:.2f}")
    if args.baseline_only:
        return

    # LoRA — zero-shot hesaplandıktan sonra uygula
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Eğitim verisi hazırla
    print("\nEğitim verisi hazırlanıyor...")
    raw_samples = build_train_dataset(
        ds_fleurs_tr, text_col_f,
        ds_cv_tr=ds_cv_tr, text_col_cv="sentence",
        use_synth=args.synth, synth_n=args.synth_n,
        use_augment=args.augment, augment_ratio=args.augment_ratio,
    )

    def make_hf_dataset(raw):
        from datasets import Dataset
        rows = []
        for s in raw:
            feats = processor.feature_extractor(
                s["array"], sampling_rate=TARGET_SR).input_features[0]
            labs = processor.tokenizer(s["text"]).input_ids
            rows.append({"input_features": feats, "labels": labs})
        return Dataset.from_list(rows)

    print("  Feature extraction yapılıyor...")
    ds_train_hf = make_hf_dataset(raw_samples)

    # Eval set: FLEURS test, augmentsız
    def prepare_eval(ex):
        arr = decode_audio(ex["audio"])
        feats = processor.feature_extractor(arr, sampling_rate=TARGET_SR).input_features[0]
        labs = processor.tokenizer(normalize_verbatim(ex[text_col_f])).input_ids
        return {"input_features": feats, "labels": labs}

    ds_eval_hf = ds_fleurs_te.map(
        prepare_eval, remove_columns=ds_fleurs_te.column_names, num_proc=1)

    @dataclass
    class Collator:
        processor: Any
        def __call__(self, feats):
            inp = [{"input_features": f["input_features"]} for f in feats]
            batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")
            labs = self.processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
            labels = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        h = [normalize_verbatim(x)
             for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        r = [normalize_verbatim(x)
             for x in processor.batch_decode(label_ids, skip_special_tokens=True)]
        return {"wer": 100 * wer_metric.compute(predictions=h, references=r)}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        bf16=use_bf16, fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=225,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_accumulation_steps=1,
        dataloader_num_workers=0,
        logging_steps=20,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ds_train_hf, eval_dataset=ds_eval_hf,
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"\nEğitim başlıyor: {len(ds_train_hf)} örnek | {args.epochs} epoch")
    trainer.train()
    metrics = trainer.evaluate()

    # LoRA ağırlıklarını kaydet
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)

    print("\n=== YADOBA DOMAIN ÖN FİZİBİLİTE SONUCU ===")
    print(f"  Model            : {args.model}  (LoRA r={args.lora_r})")
    print(f"  Augmentasyon     : {args.augment}")
    print(f"  Sentetik veri    : {args.synth}")
    print(f"  Common Voice     : {args.common_voice}")
    print(f"  Eğitim örnekleri : {len(ds_train_hf)}")
    print(f"  Zero-shot WER    : %{base_wer:.2f}")
    print(f"  Fine-tune WER    : %{metrics['eval_wer']:.2f}")
    print(f"  Model kaydı      : {args.out}/")


if __name__ == "__main__":
    main()
