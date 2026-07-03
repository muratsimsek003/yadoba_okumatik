import paramiko, sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"

HOST = "72.61.189.209"
USER = "root"
PASS = ".6XgS?zDlh?hPC7f"

def ssh(client, cmd, timeout=300):
    print(f"\n>>> {cmd[:110]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "replace").strip()
    err = stderr.read().decode("utf-8", "replace").strip()
    if out: print(out[-800:])
    if err:
        err_clean = err[-400:].encode("ascii","replace").decode("ascii")
        if "warning" not in err_clean.lower()[:30]:
            print("STDERR:", err_clean)
    return out, exit_code

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=15)
print("=== VPS baglantisi kuruldu ===")

# 0. unzip ve curl kur
print("\n--- 0. unzip/curl kontrolu ---")
ssh(client, "apt-get install -y -qq unzip curl && echo TOOLS_OK", 60)

# 1. GitHub baglantisi kontrol et
print("\n--- 1. GitHub erisimi test ---")
out, code = ssh(client, "curl -s -o /dev/null -w '%{http_code}' https://github.com/muratsimsek003/yadoba_okumatik/archive/refs/heads/main.zip --max-time 10", 20)
print(f"HTTP: {out}")

# 2. Repo indir
print("\n--- 2. Repo indiriliyor ---")
ssh(client, "curl -L --max-time 90 -o /home/yadoba.zip 'https://github.com/muratsimsek003/yadoba_okumatik/archive/refs/heads/main.zip' && echo INDIRILDI", 120)

# 3. Cikart
print("\n--- 3. Zip aciliyor ---")
ssh(client, "cd /home && unzip -q -o yadoba.zip && rm -rf /var/www/yadoba && mv /home/yadoba_okumatik-main /var/www/yadoba && rm -f /home/yadoba.zip && echo TAMAM")

# 4. Dosya kontrolu
print("\n--- 4. Dosya kontrolu ---")
ssh(client, "ls /var/www/yadoba/")

# 5. Python venv
print("\n--- 5. Virtual environment ---")
ssh(client, "cd /var/www/yadoba && python3 -m venv .venv && echo VENV_OK", 60)

# 6. CPU PyTorch
print("\n--- 6. PyTorch CPU kuruluyor (~5-10dk) ---")
ssh(client, "/var/www/yadoba/.venv/bin/pip install -q --upgrade pip", 60)
ssh(client, "/var/www/yadoba/.venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu && echo TORCH_OK", 600)

# 7. Diger bagimliliklar
print("\n--- 7. Bagimliliklar kuruluyor ---")
ssh(client, "/var/www/yadoba/.venv/bin/pip install -q fastapi 'uvicorn[standard]' python-multipart librosa soundfile transformers accelerate && echo DEPS_OK", 300)

# 8. Data klasoru
ssh(client, "mkdir -p /var/www/yadoba/data && chown -R root:root /var/www/yadoba")

# 9. Systemd service
print("\n--- 8. Systemd servis ---")
svc = "[Unit]\nDescription=YADOBA\nAfter=network.target\n\n[Service]\nType=simple\nUser=root\nWorkingDirectory=/var/www/yadoba\nExecStart=/var/www/yadoba/.venv/bin/python app.py\nRestart=always\nRestartSec=5\nEnvironment=PYTHONUNBUFFERED=1\n\n[Install]\nWantedBy=multi-user.target\n"
ssh(client, f"printf '%s' '{svc}' > /etc/systemd/system/yadoba.service")
ssh(client, "systemctl daemon-reload && systemctl enable yadoba && systemctl restart yadoba && sleep 4 && systemctl is-active yadoba")

# 10. Nginx
print("\n--- 9. Nginx ayarlaniyor ---")
ng = "server {\n    listen 80;\n    server_name _;\n    client_max_body_size 50M;\n    proxy_read_timeout 300s;\n    proxy_send_timeout 300s;\n    location / {\n        proxy_pass http://127.0.0.1:8000;\n        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n    }\n}\n"
ssh(client, f"printf '%s' '{ng}' > /etc/nginx/sites-available/yadoba")
ssh(client, "ln -sf /etc/nginx/sites-available/yadoba /etc/nginx/sites-enabled/yadoba && rm -f /etc/nginx/sites-enabled/default && nginx -t && systemctl restart nginx && echo NGINX_OK")

# 11. Son kontrol
print("\n--- 10. Son durum ---")
ssh(client, "systemctl is-active yadoba && systemctl is-active nginx")
ssh(client, "ss -tlnp | grep -E '8000|:80'")
out, _ = ssh(client, "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/login.html")
print(f"Login sayfasi HTTP: {out}")

client.close()
print("\n=== KURULUM TAMAMLANDI ===")
print(f"Site: http://{HOST}")
