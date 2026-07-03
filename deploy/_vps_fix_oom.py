import paramiko, os, time
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=60):
    print(f">>> {cmd[:90]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8","replace").strip()
    err = stderr.read().decode("utf-8","replace").strip()
    if out: print(out[-400:])
    if err: print("ERR:", err[-200:])
    return out

# 1. Swap ekle (4GB)
print("=== 1. Swap alanı ekleniyor (4GB) ===")
ssh("swapon --show")
ssh("fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo SWAP_OK", 30)
ssh("echo '/swapfile none swap sw 0 0' >> /etc/fstab")
ssh("free -h")

# 2. Güncel app.py yükle (WHISPER_MODEL env desteğiyle)
print("\n=== 2. app.py güncelleniyor ===")
sftp = client.open_sftp()
sftp.put(r"c:\Users\Murat\Desktop\yadoba-okumetrik\app.py", "/var/www/yadoba/app.py")
sftp.close()
print("app.py yüklendi.")

# 3. Systemd service'e WHISPER_MODEL=openai/whisper-small ekle
print("\n=== 3. Servis whisper-small kullanacak şekilde güncelleniyor ===")
service = """[Unit]
Description=YADOBA Okuma Degerlendirme
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/yadoba
ExecStart=/var/www/yadoba/.venv/bin/python app.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
Environment=WHISPER_MODEL=openai/whisper-small

[Install]
WantedBy=multi-user.target
"""
sftp = client.open_sftp()
with sftp.open("/etc/systemd/system/yadoba.service", "w") as f:
    f.write(service)
sftp.close()

# 4. Servisi yeniden başlat
print("\n=== 4. Servis yeniden başlatılıyor ===")
ssh("systemctl daemon-reload && systemctl restart yadoba && sleep 5 && systemctl is-active yadoba")

# 5. Log kontrol
print("\n=== 5. Log kontrolü ===")
time.sleep(3)
ssh("journalctl -u yadoba --no-pager -n 15 --output=short")

# 6. Test
print("\n=== 6. HTTP testi ===")
print(ssh('curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3001/login.html'))

print("\n=== Swap durumu ===")
ssh("free -h")

client.close()
print("\n=== TAMAMLANDI ===")
print("Whisper small model kullanılıyor (~500MB RAM)")
print("Site: https://yadoba.isteknoloji.cloud")
