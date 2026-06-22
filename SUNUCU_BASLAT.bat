@echo off
title YADOBA Sunucusu
cd /d "%~dp0"
echo.
echo  ==========================================
echo   YADOBA 2.0 - Okuma Degerlendirme Sistemi
echo  ==========================================
echo.
echo  Sunucu baslatiliyor...
echo.
echo  Tarayicinizda su adresi acin:
echo.
echo      http://localhost:8000
echo.
echo  Kapatmak icin bu pencereyi kapatin.
echo  ==========================================
echo.
.venv\Scripts\python.exe app.py
pause
