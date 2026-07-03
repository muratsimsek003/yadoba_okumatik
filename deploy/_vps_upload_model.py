import paramiko, os, time, pathlib
os.environ["PYTHONIOENCODING"] = "utf-8"

LOCAL_MODEL = r"D:\ct2_whisper_yadoba"
REMOTE_MODEL = "/var/www/yadoba/ct2_model"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8","replace").strip()

# Uzak klasör oluştur
ssh(f"mkdir -p {REMOTE_MODEL}")

sftp = client.open_sftp()
files = list(pathlib.Path(LOCAL_MODEL).rglob("*"))
files = [f for f in files if f.is_file()]
total = sum(f.stat().st_size for f in files)
done = 0
t0 = time.time()

print(f"Toplam {len(files)} dosya, {total/1024**2:.0f} MB yüklenecek...")
for f in files:
    rel = f.relative_to(LOCAL_MODEL)
    remote = f"{REMOTE_MODEL}/{str(rel).replace(chr(92), '/')}"
    # Uzak alt klasör varsa oluştur
    rdir = remote.rsplit("/", 1)[0]
    ssh(f"mkdir -p {rdir}")
    sftp.put(str(f), remote)
    done += f.stat().st_size
    elapsed = time.time() - t0
    speed = done / elapsed / 1024**2
    print(f"  [{done/1024**2:.0f}/{total/1024**2:.0f} MB] {rel} — {speed:.1f} MB/s")

sftp.close()
print(f"\nYükleme tamamlandı ({time.time()-t0:.0f}s)")

# Servis env güncelle
print("\n=== Servis güncelleniyor (yerel model) ===")
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
Environment=WHISPER_MODEL=/var/www/yadoba/ct2_model

[Install]
WantedBy=multi-user.target
"""
sftp2 = client.open_sftp()
with sftp2.open("/etc/systemd/system/yadoba.service", "w") as f:
    f.write(service)
sftp2.close()

print(ssh("systemctl daemon-reload && systemctl restart yadoba && sleep 5 && systemctl is-active yadoba"))
print(ssh("journalctl -u yadoba --no-pager -n 6 --output=short"))
print(ssh('curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3001/login.html'))

client.close()
print("\nFine-tuned model aktif: /var/www/yadoba/ct2_model")
