import paramiko, os, stat
os.environ["PYTHONIOENCODING"] = "utf-8"

HOST = "72.61.189.209"
USER = "root"
PASS = ".6XgS?zDlh?hPC7f"
LOCAL  = r"c:\Users\Murat\Desktop\yadoba-okumetrik"
REMOTE = "/var/www/yadoba"

# Yüklenecek dosya ve klasörler
UPLOAD_FILES = [
    "app.py", "login.html", "reading.html", "admin.html",
    "index.html", "requirements.txt",
]
UPLOAD_DIRS = ["okumetrik"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=15)
sftp = client.open_sftp()

def mkdir_p(sftp, remote_dir):
    parts = remote_dir.split("/")
    path = ""
    for p in parts:
        if not p: continue
        path += "/" + p
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)

def upload_dir(sftp, local_dir, remote_dir):
    mkdir_p(sftp, remote_dir)
    for item in os.listdir(local_dir):
        if item.startswith("__pycache__") or item.endswith(".pyc"):
            continue
        lp = os.path.join(local_dir, item)
        rp = remote_dir + "/" + item
        if os.path.isdir(lp):
            upload_dir(sftp, lp, rp)
        else:
            sftp.put(lp, rp)
            print(f"  {rp}")

print("=== Dosyalar yükleniyor ===")
mkdir_p(sftp, REMOTE)
mkdir_p(sftp, REMOTE + "/data")

for f in UPLOAD_FILES:
    lp = os.path.join(LOCAL, f)
    if os.path.exists(lp):
        sftp.put(lp, f"{REMOTE}/{f}")
        print(f"  {REMOTE}/{f}")
    else:
        print(f"  [ATLANDI] {f}")

for d in UPLOAD_DIRS:
    lp = os.path.join(LOCAL, d)
    if os.path.exists(lp):
        upload_dir(sftp, lp, f"{REMOTE}/{d}")
    else:
        print(f"  [ATLANDI] {d}/")

sftp.close()
print("\n=== Yükleme tamamlandı ===")

def ssh(cmd, timeout=60):
    print(f"\n>>> {cmd[:100]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8","replace").strip()
    err = stderr.read().decode("utf-8","replace").strip()
    if out: print(out[-500:])
    if err: print("ERR:", err[-300:])
    return out

# Port 3001 kullan (80 ve 8000 Coolify'da)
print("\n=== Port 3001 için app.py güncelleniyor ===")
ssh("sed -i 's/port=8000/port=3001/g' /var/www/yadoba/app.py && echo PORT_OK")

# Systemd servis durdur ve güncelle
print("\n=== Servis yeniden başlatılıyor ===")
ssh("systemctl stop yadoba && systemctl start yadoba && sleep 4 && systemctl is-active yadoba")

# Kontrol
print("\n=== Son kontrol ===")
ssh("ss -tlnp | grep 3001")
out = ssh("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3001/login.html")
print(f"Login sayfasi HTTP: {out}")

# Firewall 3001 aç
ssh("ufw allow 3001/tcp 2>/dev/null; iptables -I INPUT -p tcp --dport 3001 -j ACCEPT 2>/dev/null; echo FW_OK")

# Log
ssh("journalctl -u yadoba --no-pager -n 10")

client.close()
print(f"\n=== TAMAMLANDI ===")
print(f"Site: http://72.61.189.209:3001")
