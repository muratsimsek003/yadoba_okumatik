# setup_windows.ps1 — RTX 4070 / Windows kurulum
# Çalıştırma (proje kök klasöründe, PowerShell):
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1

Write-Host "== Python venv olusturuluyor (.venv) ==" -ForegroundColor Cyan
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "== pip guncelleniyor ==" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "== CUDA'li PyTorch kuruluyor (cu124, RTX 4070) ==" -ForegroundColor Cyan
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "== Diger bagimliliklar ==" -ForegroundColor Cyan
pip install -r pre_feasibility/requirements.txt

Write-Host "== GPU kontrolu ==" -ForegroundColor Cyan
python pre_feasibility/check_gpu.py

Write-Host "`nKurulum tamam. VS Code'da klasoru acin, sag altta .venv yorumlayicisini secin." -ForegroundColor Green
Write-Host "Sonraki adim:  python pre_feasibility/finetune_whisper_tr.py --baseline_only" -ForegroundColor Green
