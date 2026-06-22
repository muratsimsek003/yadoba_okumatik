#!/usr/bin/env bash
# setup_linux.sh — RTX 4070 / Linux veya WSL2 kurulum
#   bash setup_linux.sh
set -e

echo "== Python venv (.venv) =="
python3 -m venv .venv
source .venv/bin/activate

echo "== pip güncelleniyor =="
python -m pip install --upgrade pip

echo "== CUDA'lı PyTorch (cu124, RTX 4070) =="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "== Diğer bağımlılıklar =="
pip install -r pre_feasibility/requirements.txt

echo "== GPU kontrolü =="
python pre_feasibility/check_gpu.py

echo ""
echo "Kurulum tamam. Sonraki adım:"
echo "  python pre_feasibility/finetune_whisper_tr.py --baseline_only"
