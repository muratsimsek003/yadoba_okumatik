"""
YADOBA — Model Karsilastirma: WER + F1 Hesaplama
=================================================
FLEURS Turkce test seti uzerinde 4 modeli karsilastirir:
  1. whisper-small      (mevcut VPS modeli, baseline)
  2. whisper-medium     (fine-tune oncesi, zero-shot)
  3. whisper-large-v3 fine-tune (D:/ct2_whisper_yadoba, CT2)
  4. whisper-medium fine-tune  (training/outputs/medium-tr-yadoba-ct2 sonrasi)

Mimari:
  - HF modeller (small, medium): WhisperForConditionalGeneration + WhisperProcessor
    dogrudan numpy array kabul eder, torchcodec/ffmpeg gerekmez
  - CT2 modeller (large-v3-ft, medium-ft): faster-whisper 1.1.x
    soundfile ile WAV yazilip gecilir, torchcodec gerekmez

Kullanim:
  python training/evaluate_wer.py --max_samples 200
  python training/evaluate_wer.py --models whisper-small whisper-large-v3-ft
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache\datasets")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Metin normalizasyonu ───────────────────────────────────────────────────────
def normalize_tr(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── WER + Word-F1 ─────────────────────────────────────────────────────────────
def compute_wer_f1(references: list[str], hypotheses: list[str]) -> dict:
    import jiwer
    norm_refs = [normalize_tr(r) for r in references]
    norm_hyps = [normalize_tr(h) for h in hypotheses]

    wer_val = jiwer.wer(norm_refs, norm_hyps)
    cer_val = jiwer.cer(norm_refs, norm_hyps)

    precisions, recalls, f1s = [], [], []
    for ref, hyp in zip(norm_refs, norm_hyps):
        rw = set(ref.split()); hw = set(hyp.split())
        if not rw and not hw:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0); continue
        tp   = len(rw & hw)
        prec = tp / len(hw) if hw else 0.0
        rec  = tp / len(rw) if rw else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    return {
        "wer":            round(wer_val * 100, 2),
        "cer":            round(cer_val * 100, 2),
        "word_precision": round(sum(precisions) / len(precisions) * 100, 2),
        "word_recall":    round(sum(recalls)    / len(recalls)    * 100, 2),
        "word_f1":        round(sum(f1s)        / len(f1s)        * 100, 2),
        "n_samples":      len(references),
    }


# ── HF model transkripsiyon (WhisperForConditionalGeneration) ─────────────────
def transcribe_hf(model_id: str, samples: list, batch_size: int = 8) -> list[str]:
    """
    WhisperForConditionalGeneration ile direkt inference.
    Numpy array -> feature extractor -> model.generate() -> tokenizer.decode()
    Torchcodec veya ffmpeg gerektirmez.
    """
    import torch, numpy as np
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"  HF model yukleniyor: {model_id} @ {device} ({dtype})")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()

    forced_ids = processor.get_decoder_prompt_ids(language="tr", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_ids

    transcripts = []
    total = len(samples)
    for i in range(0, total, batch_size):
        batch  = samples[i:i + batch_size]
        decoded_audios = [decode_audio_sample(s["audio"]) for s in batch]
        arrays = [a for a, _ in decoded_audios]
        sr     = 16000  # decode_audio_sample garantili 16kHz

        inputs = processor(
            arrays, sampling_rate=sr, return_tensors="pt", padding=True
        ).to(device)
        if dtype == torch.float16:
            inputs["input_features"] = inputs["input_features"].half()

        with torch.no_grad():
            ids = model.generate(**inputs, forced_decoder_ids=forced_ids)
        decoded = processor.batch_decode(ids, skip_special_tokens=True)
        transcripts.extend(t.strip() for t in decoded)

        done = min(i + batch_size, total)
        if done % 50 == 0 or done == total:
            print(f"    {done}/{total} tamamlandi")

    del model, processor
    try:
        import gc; torch.cuda.empty_cache(); gc.collect()
    except Exception:
        pass
    return transcripts


# ── CT2 model transkripsiyon (faster-whisper 1.1.x) ──────────────────────────
def transcribe_ct2(model_path: str, samples: list) -> list[str]:
    """
    faster-whisper 1.1.x ile CTranslate2 model inference.
    Numpy array'i soundfile ile gecici WAV'a yazar, path gecer.
    Torchcodec gerektirmez (soundfile kullanir).
    """
    import torch, numpy as np, soundfile as sf, tempfile
    from faster_whisper import WhisperModel

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"  CT2 model yukleniyor: {model_path} @ {device} ({compute_type})")
    fw = WhisperModel(model_path, device=device, compute_type=compute_type)

    transcripts = []
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for i, s in enumerate(samples):
            audio, sr = decode_audio_sample(s["audio"])
            wav_p = tmp_dir / f"tmp_{i}.wav"
            sf.write(str(wav_p), audio, sr)

            segs, _ = fw.transcribe(
                str(wav_p), language="tr", task="transcribe",
                temperature=0.0, beam_size=5,
            )
            transcripts.append(" ".join(seg.text.strip() for seg in segs).strip())
            wav_p.unlink(missing_ok=True)

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                print(f"    {i+1}/{len(samples)} tamamlandi")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    del fw
    try:
        import gc; torch.cuda.empty_cache(); gc.collect()
    except Exception:
        pass
    return transcripts


# ── Audio decode (torchcodec bypass) ─────────────────────────────────────────
def decode_audio_sample(raw_audio: dict) -> tuple:
    """
    datasets Audio(decode=False) ile alinan ham veriyi soundfile ile decode eder.
    torchcodec veya ffmpeg gerekmez.
    Returns: (audio_array float32, sample_rate int)
    """
    import io, numpy as np, soundfile as sf
    from scipy.signal import resample as scipy_resample

    if raw_audio.get("bytes"):
        audio, sr = sf.read(io.BytesIO(raw_audio["bytes"]))
    else:
        audio, sr = sf.read(raw_audio["path"])

    if audio.ndim > 1:
        audio = audio.mean(axis=1)   # stereo -> mono

    audio = np.array(audio, dtype=np.float32)

    if sr != 16000:
        target_len = int(len(audio) * 16000 / sr)
        audio = scipy_resample(audio, target_len).astype(np.float32)
        sr = 16000

    return audio, sr


# ── FLEURS veri seti ──────────────────────────────────────────────────────────
def load_fleurs_test(max_samples: Optional[int] = None):
    from datasets import load_dataset, Audio
    print("FLEURS Turkce test seti yukleniyor (google/fleurs, tr_tr)...")
    ds = load_dataset("google/fleurs", "tr_tr", split="test")
    # decode=False: ham bytes alir, torchcodec tetiklenmez
    ds = ds.cast_column("audio", Audio(decode=False))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  {len(ds)} ornek yuklendi")
    return ds


# ── Model tanimlamalari ────────────────────────────────────────────────────────
MODELS = {
    "whisper-small": {
        "label":    "Whisper Small (mevcut VPS, baseline)",
        "category": "baseline",
        "backend":  "hf",
        "model_id": "openai/whisper-small",
    },
    "whisper-medium-zero": {
        "label":    "Whisper Medium (zero-shot)",
        "category": "zero-shot",
        "backend":  "hf",
        "model_id": "openai/whisper-medium",
    },
    "whisper-medium-ft": {
        "label":    "Whisper Medium (YADOBA fine-tune)",
        "category": "fine-tuned",
        "backend":  "ct2",
        "model_id": r"D:\yadoba-training\medium-tr-yadoba-ct2",
    },
    "whisper-large-v3-ft": {
        "label":    "Whisper Large-v3 (LoRA fine-tune, chk-196)",
        "category": "fine-tuned",
        "backend":  "ct2",
        "model_id": r"D:\ct2_whisper_yadoba",
    },
}


def evaluate_model(key: str, samples: list) -> dict:
    info    = MODELS[key]
    model_id = info["model_id"]
    backend  = info["backend"]
    refs     = [s["raw_transcription"] for s in samples]

    if backend == "ct2" and not Path(model_id).exists():
        return {**info, "skipped": True, "reason": f"Model yolu yok: {model_id}"}

    print(f"\n{'='*60}")
    print(f"Model:   {info['label']}")
    print(f"Path:    {model_id}")
    print(f"Backend: {backend}")
    t0 = time.time()

    try:
        if backend == "hf":
            hyps = transcribe_hf(model_id, samples)
        else:
            hyps = transcribe_ct2(model_id, samples)
    except Exception as e:
        import traceback; traceback.print_exc()
        return {**info, "skipped": True, "reason": str(e)}

    metrics = compute_wer_f1(refs, hyps)
    elapsed = round(time.time() - t0, 1)
    print(f"  WER: {metrics['wer']}%  |  Word-F1: {metrics['word_f1']}%  |  Sure: {elapsed}s")
    return {**info, **metrics, "eval_time_sec": elapsed, "skipped": False}


def print_table(results: list[dict]):
    print("\n" + "=" * 82)
    print(f"{'Model':<42} {'WER%':>6} {'CER%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6}")
    print("-" * 82)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['label']:<40} [ATLANDI: {r.get('reason','')[:50]}]")
        else:
            print(f"  {r['label']:<40} {r['wer']:>6.1f} {r['cer']:>6.1f} "
                  f"{r['word_precision']:>6.1f} {r['word_recall']:>6.1f} {r['word_f1']:>6.1f}")
    print("=" * 82)
    best = min((r for r in results if not r.get("skipped")), key=lambda x: x["wer"], default=None)
    if best:
        print(f"  En dusuk WER: {best['label']} ({best['wer']}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--models", nargs="+",
                    choices=list(MODELS.keys()) + ["all"], default=["all"])
    args = ap.parse_args()

    ds      = load_fleurs_test(max_samples=args.max_samples)
    samples = list(ds)
    keys    = list(MODELS.keys()) if "all" in args.models else args.models

    results = []
    for key in keys:
        try:
            r = evaluate_model(key, samples)
        except Exception as e:
            r = {**MODELS[key], "skipped": True, "reason": str(e)}
        results.append(r)

    print_table(results)

    out_json = RESULTS_DIR / "wer_comparison.json"
    out_txt  = RESULTS_DIR / "wer_comparison.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"YADOBA Model Karsilastirmasi — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"FLEURS Turkce test ({len(samples)} ornek)\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            if not r.get("skipped"):
                f.write(f"[{r['category'].upper()}] {r['label']}\n")
                f.write(f"  WER:            {r['wer']}%\n")
                f.write(f"  CER:            {r['cer']}%\n")
                f.write(f"  Word-Precision: {r['word_precision']}%\n")
                f.write(f"  Word-Recall:    {r['word_recall']}%\n")
                f.write(f"  Word-F1:        {r['word_f1']}%\n\n")
            else:
                f.write(f"[ATLANDI] {r['label']}: {r.get('reason','')}\n\n")

    print(f"\nSonuclar kaydedildi:\n  {out_json}\n  {out_txt}")


if __name__ == "__main__":
    main()
