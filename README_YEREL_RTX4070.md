# Yerel Kurulum — RTX 4070 (12 GB) + VS Code

Bu rehber, ön fizibilite (Türkçe ASR fine-tune + WER) çalışmasını **kendi
bilgisayarında** çalıştırmak içindir. Bu sandbox'ta GPU/HF erişimi yoktu; senin
makinende ikisi de var, yani her şey gerçek çalışır.

## Önkoşullar
- NVIDIA RTX 4070 (12 GB) + güncel **NVIDIA sürücüsü** (CUDA 12.x destekli)
- Python 3.10–3.12
- VS Code + **Python** ve **Pylance** eklentileri
- Disk: model+veri için ~10–15 GB boş alan

> CUDA Toolkit'i ayrıca kurmana gerek yok; pip'in CUDA'lı PyTorch tekerleği
> gerekli runtime'ı içerir. Sadece güncel NVIDIA sürücüsü yeterli.

## Kurulum (tek komut)

**Windows (PowerShell, proje kök klasöründe):**
```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```
**Linux / WSL2:**
```bash
bash setup_linux.sh
```

Bu betik: `.venv` oluşturur, **CUDA'lı PyTorch (cu124)** kurar, bağımlılıkları
yükler ve `check_gpu.py` ile kurulumu doğrular.

### Manuel kurulum (isteğe bağlı)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r pre_feasibility/requirements.txt
python pre_feasibility/check_gpu.py
```

## VS Code'da çalıştırma
1. Klasörü VS Code'da aç (`File > Open Folder`).
2. Sağ altta veya `Ctrl+Shift+P > Python: Select Interpreter` ile **.venv**'i seç.
3. **Run and Debug** (Ctrl+Shift+D) panelinde hazır profiller var:
   - **1) GPU kontrol**
   - **2) Zero-shot WER (hızlı)**
   - **3) Fine-tune whisper-small (RTX 4070)**
   - **4) Fine-tune whisper-medium (LoRA, 12GB)**
   - **OKUMETRİK demo + testler**
   Birini seçip ▶ ile çalıştır.

Ya da terminalden:
```powershell
python pre_feasibility/check_gpu.py
python pre_feasibility/finetune_whisper_tr.py --baseline_only
python pre_feasibility/finetune_whisper_tr.py --model openai/whisper-small --epochs 3 --batch_size 16
```

## RTX 4070 (12 GB) için ayarlar

| Model | Komut | ~VRAM | Durum |
|---|---|---|---|
| whisper-small | `--model openai/whisper-small --batch_size 16` | ~8–9 GB | ✅ rahat |
| whisper-medium | `--model openai/whisper-medium --lora --grad_ckpt --batch_size 4 --grad_accum 4` | ~10–11 GB | ✅ sınırda |
| whisper-large-v3 | `--lora --grad_ckpt --batch_size 1 --grad_accum 8` | >12 GB | ⚠ yavaş/riskli |

- **bf16 otomatik açılır** (Ada mimarisi destekler) — hız + bellek avantajı.
- VRAM yetmezse: `--batch_size` düşür, `--grad_accum` artır, `--grad_ckpt` ekle.
- Süre/maliyet: `--max_train` ve `--max_eval` ile veri miktarını sınırla (PoC için 1500/400 yeterli).

## Windows notları
- `dataloader` çoklu-işlem Windows'ta sorun çıkarırsa: `--num_workers 0`.
- Türkçe karakterler için terminal UTF-8: ayarlar zaten `PYTHONUTF8=1` veriyor.
- Common Voice kullanacaksan: `huggingface-cli login` (token) ve veri şartlarını kabul et;
  FLEURS (varsayılan) bunu gerektirmez.

## Beklenen çıktı
```
[GPU] NVIDIA GeForce RTX 4070 | VRAM ~12.0 GB
[precision] bf16=True fp16=False
  >> ZERO-SHOT (fine-tune'suz) WER = %1X.XX
...
=== ÖN FİZİBİLİTE SONUCU ===
  Zero-shot WER    : %1X.XX
  Fine-tune sonrası: %X.XX
```

## Hatırlatma (strateji)
WER bir **destek** kanıtıdır (akustik altyapı çalışıyor). Başvurunun asıl yenilik
kanıtı `okumetrik` motorunun ürettiği **miscue F1**'dir. İkisini birlikte sun.
Ayrıntı için ana `README.md` ve `pre_feasibility/README.md`.
