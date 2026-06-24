#!/bin/bash
# YADOBA VPS Kurulum Scripti (Ubuntu 22.04 / Hostinger VPS)
# Çalıştır: bash setup_vps.sh

set -e
echo "=== YADOBA VPS Kurulumu ==="

# 1. Sistem güncellemesi
apt-get update -y && apt-get upgrade -y

# 2. Temel araçlar
apt-get install -y python3 python3-pip python3-venv git nginx ffmpeg curl

# 3. Python sürüm kontrolü
python3 --version

# 4. Uygulama klasörü
mkdir -p /var/www/yadoba
cd /var/www/yadoba

# 5. GitHub'dan clone (URL'yi değiştirin gerekirse)
if [ ! -d ".git" ]; then
    git clone https://github.com/muratsimsek003/yadoba_okumatik.git .
else
    git pull origin main
fi

# 6. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 7. CPU-only PyTorch (GPU yoksa — daha hızlı kurulum)
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 8. Data klasörü
mkdir -p data

# 9. Systemd servis kopyala
cp deploy/yadoba.service /etc/systemd/system/yadoba.service
systemctl daemon-reload
systemctl enable yadoba
systemctl start yadoba

# 10. Nginx konfigürasyonu
cp deploy/nginx.conf /etc/nginx/sites-available/yadoba
ln -sf /etc/nginx/sites-available/yadoba /etc/nginx/sites-enabled/yadoba
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

echo ""
echo "=== Kurulum Tamamlandı ==="
echo "Uygulama çalışıyor: http://$(curl -s ifconfig.me)"
echo ""
echo "Servis durumu: systemctl status yadoba"
echo "Loglar:        journalctl -u yadoba -f"
