#!/bin/bash
# Güncelleme scripti — yeni sürüm çıkınca çalıştırın
cd /var/www/yadoba
git pull origin main
systemctl restart yadoba
echo "Güncellendi ve yeniden başlatıldı."
