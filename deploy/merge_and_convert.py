"""
LoRA adaptörünü base model ile birleştir ve CTranslate2 int8 formatına dönüştür.
Sonuç: D:/ct2_whisper_yadoba/
"""
import os, time, subprocess, sys
os.environ["HF_HOME"] = "D:/hf_cache"

ADAPTER_PATH = "D:/hf_cache/whisper-child-tr-poc/checkpoint-196"
MERGED_PATH  = "D:/merged_whisper_yadoba"
CT2_PATH     = "D:/ct2_whisper_yadoba"

print("=== 1. Base model + LoRA merge ===")
t0 = time.time()

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device}")

print("Base model yukleniyor (float16)...")
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,
    cache_dir="D:/hf_cache"
)
base = base.to(device)
print(f"Base model yuklendi ({time.time()-t0:.0f}s)")

print("LoRA adapter yukleniyor (checkpoint-196)...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
print("Merge & unload...")
merged = model.merge_and_unload()
merged = merged.to(torch.float16)
print(f"Merge tamamlandi ({time.time()-t0:.0f}s)")

print(f"Kaydediliyor -> {MERGED_PATH}")
merged.save_pretrained(MERGED_PATH, safe_serialization=True)
WhisperProcessor.from_pretrained("D:/hf_cache/whisper-child-tr-poc").save_pretrained(MERGED_PATH)
print(f"Kayit tamam ({time.time()-t0:.0f}s)")

print("\n=== 2. CTranslate2 int8 donusturme ===")
result = subprocess.run(
    [sys.executable, "-m", "ctranslate2.tools.whisper",
     "--model", MERGED_PATH,
     "--output_dir", CT2_PATH,
     "--quantization", "int8",
     "--force"],
    capture_output=True, text=True, encoding="utf-8"
)
# Alternatif komut satırı
if result.returncode != 0:
    result = subprocess.run(
        ["ct2-transformers-converter",
         "--model", MERGED_PATH,
         "--output_dir", CT2_PATH,
         "--quantization", "int8",
         "--force"],
        capture_output=True, text=True, encoding="utf-8"
    )

print(result.stdout[-600:] if result.stdout else "(cikti yok)")
if result.stderr: print("STDERR:", result.stderr[-300:])
print(f"Donusum: {'BASARILI' if result.returncode==0 else 'BASARISIZ (kod='+str(result.returncode)+')'}")
print(f"Toplam sure: {time.time()-t0:.0f}s")

if result.returncode == 0:
    import pathlib
    files = list(pathlib.Path(CT2_PATH).rglob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024**2
    print(f"Model boyutu: {total_mb:.0f} MB -> {CT2_PATH}")
