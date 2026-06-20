#!/usr/bin/env python3
"""
finetune_whisper_tr.py — Türkçe ASR ön fizibilitesi: fine-tune + WER.
RTX 4070 (12 GB VRAM, Ada) + yerel tek GPU için ayarlanmıştır.

HIZLI KULLANIM (VS Code terminali / PowerShell):
  # 1) en hızlı: fine-tune YAPMADAN zero-shot WER (dakikalar)
  python finetune_whisper_tr.py --baseline_only

  # 2) tam PoC: whisper-small full fine-tune (12 GB'a rahat sığar)
  python finetune_whisper_tr.py --model openai/whisper-small --epochs 3 --batch_size 16

  # 3) daha güçlü: whisper-medium (LoRA + gradient checkpointing ile 12 GB'a sığar)
  python finetune_whisper_tr.py --model openai/whisper-medium --lora --grad_ckpt --batch_size 4 --grad_accum 4

VRAM rehberi (RTX 4070 12 GB):
  whisper-small  full FT, bs16, bf16            -> ~8-9 GB   ✅ rahat
  whisper-medium LoRA + grad_ckpt, bs4, accum4  -> ~10-11 GB ✅ sınırda
  whisper-large-v3 LoRA, bs1-2, grad_ckpt       -> >12 GB    ⚠ riskli/yavaş
"""
import argparse
import io
import logging
import warnings
import numpy as np

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def decode_audio(audio_item, target_sr=16000):
    """OGG/MP3/WAV baytlarını librosa ile numpy dizisine çevir.

    datasets>=3.x audio decoder olarak torchcodec istiyor; Windows'ta
    FFmpeg DLL'leri olmadan çalışmıyor. Bu fonksiyon datasets Audio
    decoder'ını bypass ederek librosa → audioread → Windows MF zincirini
    kullanır (FFmpeg kurulu olmasa da Windows 10/11'de OGG desteklenir).
    """
    import librosa
    raw = audio_item.get("bytes") or audio_item.get("path")
    if isinstance(raw, bytes):
        array, _ = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
    else:
        array, _ = librosa.load(raw, sr=target_sr, mono=True)
    return array.astype(np.float32)


def load_tr_dataset(name, max_train, max_eval):
    from datasets import load_dataset, Audio
    if name == "fleurs":
        ds_tr = load_dataset("google/fleurs", "tr_tr", split="train")
        ds_te = load_dataset("google/fleurs", "tr_tr", split="test")
        text_col = "transcription"
    elif name == "common_voice":
        ds_tr = load_dataset("mozilla-foundation/common_voice_17_0", "tr",
                             split="train", trust_remote_code=True)
        ds_te = load_dataset("mozilla-foundation/common_voice_17_0", "tr",
                             split="test", trust_remote_code=True)
        text_col = "sentence"
    else:
        raise ValueError(name)
    # decode=False: ham baytları al; ses çözme decode_audio() ile yapılır
    # (torchcodec yerine librosa kullanmak için)
    ds_tr = ds_tr.cast_column("audio", Audio(decode=False))
    ds_te = ds_te.cast_column("audio", Audio(decode=False))
    if max_train:
        ds_tr = ds_tr.select(range(min(max_train, len(ds_tr))))
    if max_eval:
        ds_te = ds_te.select(range(min(max_eval, len(ds_te))))
    return ds_tr, ds_te, text_col


def normalize_tr(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower().strip()


def main():
    ap = argparse.ArgumentParser(description="Türkçe Whisper fine-tune + WER (RTX 4070)")
    ap.add_argument("--dataset", default="fleurs", choices=["fleurs", "common_voice"])
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--lora", action="store_true", help="LoRA (medium/large için önerilir)")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (VRAM tasarrufu)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_train", type=int, default=1500)
    ap.add_argument("--max_eval", type=int, default=400)
    ap.add_argument("--num_workers", type=int, default=2, help="Windows'ta 0 önerilir")
    ap.add_argument("--baseline_only", action="store_true", help="Sadece zero-shot WER")
    ap.add_argument("--out", default="whisper-tr-poc")
    args = ap.parse_args()

    import torch, evaluate
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from dataclasses import dataclass
    from typing import Any

    # ---- GPU / precision ----
    if not torch.cuda.is_available():
        print("[UYARI] CUDA bulunamadı. CUDA'lı PyTorch kurun:\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cu124")
    else:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {name}  |  VRAM ~{vram:.1f} GB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # Ada -> True
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"[precision] bf16={use_bf16} fp16={use_fp16}")

    processor = WhisperProcessor.from_pretrained(args.model, language="turkish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.generation_config.language = "turkish"
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None
    if args.grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    ds_tr, ds_te, text_col = load_tr_dataset(args.dataset, args.max_train, args.max_eval)
    wer_metric = evaluate.load("wer")

    # ---- zero-shot WER ----
    def zero_shot_wer():
        model.eval(); refs, hyps = [], []
        for ex in ds_te:
            feats = processor.feature_extractor(
                decode_audio(ex["audio"]), sampling_rate=16000,
                return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=225, language="tr", task="transcribe")
            hyps.append(normalize_tr(processor.batch_decode(ids, skip_special_tokens=True)[0]))
            refs.append(normalize_tr(ex[text_col]))
        return 100 * wer_metric.compute(predictions=hyps, references=refs)

    print(f"\n[{args.model} | {args.dataset}] zero-shot WER hesaplanıyor ({len(ds_te)} örnek)...")
    base = zero_shot_wer()
    print(f"  >> ZERO-SHOT (fine-tune'suz) WER = %{base:.2f}")
    if args.baseline_only:
        return

    # ---- preprocessing ----
    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            decode_audio(audio), sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(normalize_tr(batch[text_col])).input_ids
        return batch

    ds_tr = ds_tr.map(prepare, remove_columns=ds_tr.column_names, num_proc=1)
    ds_te_p = ds_te.map(prepare, remove_columns=ds_te.column_names, num_proc=1)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                         target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

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
        h = [normalize_tr(x) for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        r = [normalize_tr(x) for x in processor.batch_decode(label_ids, skip_special_tokens=True)]
        return {"wer": 100 * wer_metric.compute(predictions=h, references=r)}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_ckpt,
        learning_rate=args.lr, warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        bf16=use_bf16, fp16=use_fp16,
        predict_with_generate=True, generation_max_length=225,
        eval_strategy="epoch", save_strategy="epoch",
        eval_accumulation_steps=1,          # tahminleri parça parça topla (VRAM)
        dataloader_num_workers=args.num_workers,
        logging_steps=25, report_to="none",
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False)

    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_te_p,
        data_collator=Collator(processor), compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor)

    trainer.train()
    metrics = trainer.evaluate()
    print("\n=== ÖN FİZİBİLİTE SONUCU ===")
    print(f"  Model            : {args.model}  (LoRA={args.lora})")
    print(f"  Zero-shot WER    : %{base:.2f}")
    print(f"  Fine-tune sonrası: %{metrics['eval_wer']:.2f}")
    print(f"  Model kaydı      : {args.out}/")


if __name__ == "__main__":
    main()
