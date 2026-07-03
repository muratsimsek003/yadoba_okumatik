import paramiko, os, time
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8","replace").strip()
    err = stderr.read().decode("utf-8","replace").strip()
    return (out + "\n" + err).strip()

# 1. Güncel app.py yükle
print("=== app.py yükleniyor ===")
sftp = client.open_sftp()
sftp.put(r"c:\Users\Murat\Desktop\yadoba-okumetrik\app.py", "/var/www/yadoba/app.py")
sftp.close()
print("Yüklendi.")

# 2. Servis dosyasını PORT=3001 ekleyerek güncelle
print("\n=== Servis güncelleniyor (PORT=3001, WHISPER_MODEL=small) ===")
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
Environment=PORT=3001
Environment=WHISPER_MODEL=openai/whisper-small

[Install]
WantedBy=multi-user.target
"""
sftp = client.open_sftp()
with sftp.open("/etc/systemd/system/yadoba.service", "w") as f:
    f.write(service)
sftp.close()
print("Servis dosyası güncellendi.")

# 3. Yeniden başlat
print("\n=== Servis yeniden başlatılıyor ===")
print(ssh("systemctl daemon-reload && systemctl restart yadoba"))
time.sleep(8)

print("\n=== Servis durumu ===")
print(ssh("systemctl is-active yadoba"))

# 4. Loglar
print("\n=== Son loglar ===")
print(ssh("journalctl -u yadoba --no-pager -n 15 --output=short"))

# 5. HTTP testi
print("\n=== HTTP testi (port 3001) ===")
print(ssh('curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3001/login.html'))

client.close()
