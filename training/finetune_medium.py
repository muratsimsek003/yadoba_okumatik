"""
YADOBA — Whisper-Medium LoRA Fine-tune (FLEURS Türkçe)
=======================================================
whisper-medium modelini FLEURS Türkçe eğitim seti üzerinde LoRA ile ince ayar yapar.

Çalıştırmak için (RTX 4070, ~3-4 saat):
  cd c:/Users/Murat/Desktop/yadoba-okumetrik
  .venv/Scripts/python.exe training/finetune_medium.py

Çıktılar:
  training/outputs/medium-tr-yadoba/        -> HuggingFace format (adapter weights)
  training/outputs/medium-tr-yadoba-merged/ -> Merged model (full weights)
  training/outputs/medium-tr-yadoba-ct2/    -> CTranslate2 int8 (faster-whisper için)

Teknik detaylar:
  - Baz model: openai/whisper-medium (~400M parametre)
  - LoRA rank: 64, alpha: 128, hedef: q_proj + v_proj + out_proj + fc1 + fc2
  - Dataset: google/fleurs tr_tr (~3000 train, ~600 val örneği)
  - Batch: 8 + gradient accumulation 4 = effective batch 32
  - Steps: 3000 (~5 epoch), learning rate: 1e-3 (cosine)
  - fp16 precision
"""

import json, os, re, sys
from pathlib import Path

# HuggingFace cache'i D: diskine yönlendir (C: dolu)
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache\datasets")
from dataclasses import dataclass

BASE_DIR  = Path(__file__).parent.parent
OUT_DIR   = Path(r"D:\yadoba-training")   # C: dolu (0.5GB), optimizer state D:'ye kaydedilir
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME       = "openai/whisper-medium"
LORA_OUT         = OUT_DIR / "medium-tr-yadoba"
MERGED_OUT       = OUT_DIR / "medium-tr-yadoba-merged"
CT2_OUT          = OUT_DIR / "medium-tr-yadoba-ct2"

# ── Türkçe metin normalizasyonu ────────────────────────────────────────────────
def normalize_tr(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Veri hazırlama ─────────────────────────────────────────────────────────────
def prepare_dataset(processor, max_train: int = None, max_val: int = None):
    from datasets import load_dataset, Audio
    import torch, io, numpy as np, soundfile as sf

    print("FLEURS Turkce yukleniyor...")
    train_ds = load_dataset("google/fleurs", "tr_tr", split="train")
    val_ds   = load_dataset("google/fleurs", "tr_tr", split="validation")

    # decode=False: torchcodec bypass, ham bytes alinir
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    val_ds   = val_ds.cast_column("audio",   Audio(decode=False))

    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_val:
        val_ds = val_ds.select(range(min(max_val, len(val_ds))))

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    def decode_audio(raw_audio):
        if raw_audio.get("bytes"):
            audio, sr = sf.read(io.BytesIO(raw_audio["bytes"]))
        else:
            audio, sr = sf.read(raw_audio["path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.array(audio, dtype=np.float32), sr

    def preprocess(batch):
        audio_arr, sr = decode_audio(batch["audio"])
        inputs = processor.feature_extractor(
            audio_arr, sampling_rate=sr, return_tensors="pt"
        )
        batch["input_features"] = inputs.input_features[0]
        label = processor.tokenizer(normalize_tr(batch["raw_transcription"]))
        batch["labels"] = label.input_ids
        return batch

    print("Egitim verisi isleniyor...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names,
                            num_proc=1, desc="Train")
    val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names,
                          num_proc=1,   desc="Val")
    return train_ds, val_ds


# ── Data collator ──────────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        import torch
        # Feature tensors
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # Label padding (-100)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # decoder_start_token_id satırını sil (shift yapılacak)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ── Custom Trainer (transformers 5.x + peft uyumlu) ───────────────────────────
def build_trainer_class():
    """
    Seq2SeqTrainer transformers 5.x + peft 0.19 ile WhisperDecoder'a
    'input_ids' iki kez geciriyor. Duzeltime: sade Trainer + manual loss.
    """
    from transformers import Trainer

    class WhisperLoRATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    return WhisperLoRATrainer


# ── Ana egitim fonksiyonu ──────────────────────────────────────────────────────
def finetune():
    import torch
    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Model + Processor ────────────────────────────────────────────────────
    print(f"\n1. Baz model yukleniyor: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    model.config.use_cache          = False   # gradient checkpointing gerektirir

    # ── LoRA ─────────────────────────────────────────────────────────────────
    print("\n2. LoRA konfigurasyonu uygulanıyor...")
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()   # peft + gradient_checkpointing icin gerekli
    model.print_trainable_parameters()

    # Whisper patch: PeftModelForSeq2SeqLM.forward() 'input_ids' bekliyor
    # ama Whisper 'input_features' kullanir. input_ids=None **kwargs uzerinden
    # WhisperDecoder'a gecip "multiple values" hatasina neden olur.
    import types

    def _whisper_peft_forward(self, input_features=None, labels=None, **kwargs):
        return self.base_model(input_features=input_features, labels=labels, **kwargs)

    model.forward = types.MethodType(_whisper_peft_forward, model)

    # ── Veri hazırlama ────────────────────────────────────────────────────────
    print("\n3. Veri seti hazirlaniyor...")
    train_ds, val_ds = prepare_dataset(processor)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Egitim argumanlari ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(LORA_OUT),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        max_steps=3000,
        gradient_checkpointing=True,
        fp16=(device == "cuda"),
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["none"],
        push_to_hub=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,       # collator'dan gelen sutunlari koru
        label_names=["labels"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    WhisperLoRATrainer = build_trainer_class()
    print("\n4. Trainer hazirlaniyor...")
    trainer = WhisperLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    # ── Egitim ───────────────────────────────────────────────────────────────
    print("\n5. Egitim basliyor...")
    print(f"   Output: {LORA_OUT}")
    print(f"   Adim sayisi: {training_args.max_steps}")
    print(f"   RTX 4070 icin tahmini sure: 3-4 saat")
    trainer.train()

    # ── LoRA adapter kaydet ───────────────────────────────────────────────────
    print("\n6. LoRA adapter kaydediliyor...")
    model.save_pretrained(str(LORA_OUT))
    processor.save_pretrained(str(LORA_OUT))
    print(f"   Kaydedildi: {LORA_OUT}")


def merge_and_convert():
    """LoRA'yı birleştir ve CTranslate2'ye dönüştür."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    if not LORA_OUT.exists():
        print(f"HATA: LoRA çıktısı bulunamadı: {LORA_OUT}")
        print("Önce finetune() çalıştırın.")
        return

    print(f"\n7. LoRA + Base merge ediliyor...")
    print(f"   Base: {MODEL_NAME}")
    print(f"   LoRA: {LORA_OUT}")

    processor = WhisperProcessor.from_pretrained(str(LORA_OUT))
    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, str(LORA_OUT))
    merged = model.merge_and_unload()
    merged = merged.to(torch.float32)

    print(f"   Kaydediliyor: {MERGED_OUT}")
    MERGED_OUT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_OUT))
    processor.save_pretrained(str(MERGED_OUT))
    print("   Merge tamamlandı")

    # CTranslate2 dönüşümü
    print(f"\n8. CTranslate2 int8 dönüşümü...")
    print(f"   Hedef: {CT2_OUT}")
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "ct2-transformers-converter",
        "--model", str(MERGED_OUT),
        "--output_dir", str(CT2_OUT),
        "--quantization", "int8",
        "--force",
    ], capture_output=True, text=True)

    if result.returncode == 0:
        # preprocessor_config.json (80 mel bin — medium için)
        import json as _json
        ppc = {
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 80,
            "hop_length": 160,
            "n_fft": 400,
            "num_mel_bins": 80,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": 16000,
        }
        (CT2_OUT / "preprocessor_config.json").write_text(
            _json.dumps(ppc, indent=2), encoding="utf-8"
        )
        size_mb = sum(f.stat().st_size for f in CT2_OUT.rglob("*") if f.is_file()) / 1024**2
        print(f"   CTranslate2 dönüşümü BAŞARILI ({size_mb:.0f} MB)")
        print(f"   Model: {CT2_OUT}")
    else:
        print(f"   HATA: {result.stderr[-500:]}")
        return False
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Whisper-Medium LoRA Fine-tune")
    ap.add_argument("--only-eval", action="store_true",
                    help="Sadece merge+convert (eğitim yok, mevcut checkpoint kullan)")
    ap.add_argument("--skip-merge", action="store_true",
                    help="Merge ve CTranslate2 dönüşümünü atla")
    args = ap.parse_args()

    if not args.only_eval:
        finetune()

    if not args.skip_merge:
        ok = merge_and_convert()
        if ok:
            print("\n" + "="*60)
            print("Fine-tune tamamlandı!")
            print(f"CTranslate2 modeli: {CT2_OUT}")
            print("\nWER değerlendirmesi için:")
            print(f"  python training/evaluate_wer.py --models whisper-small whisper-medium-zero whisper-medium-ft")
            print("\nVPS'e yüklemek için:")
            print(f"  WHISPER_MODEL={CT2_OUT} olarak systemd servisini güncelleyin")


if __name__ == "__main__":
    main()
