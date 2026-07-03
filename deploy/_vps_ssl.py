import paramiko, os, time
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8","replace").strip()
    err = stderr.read().decode("utf-8","replace").strip()
    return (out + "\n" + err).strip()

# Önce HTTP ile erişim test et
print("=== HTTP erişim testi ===")
print(ssh('curl -s -o /dev/null -w "%{http_code}" -H "Host: yadoba.isteknoloji.cloud" http://72.61.189.209/login.html'))

# Eski ACME cache'ini temizle (yeni cert alması için)
print("\n=== Eski SSL cache temizleniyor ===")
print(ssh("cat /data/coolify/proxy/acme.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print('Mevcut certler:', list(d.keys()))\" 2>/dev/null || echo 'acme.json bos'"))

# Traefik yeniden başlat
print("\n=== Traefik yeniden baslatiliyor ===")
print(ssh("docker restart coolify-proxy && echo RESTARTED", 30))

print("\n=== 10 saniye bekleniyor (cert alinmasi) ===")
time.sleep(10)

# SSL kontrol
print("\n=== SSL sertifika kontrolu ===")
print(ssh('curl -s -o /dev/null -w "%{http_code}" https://yadoba.isteknoloji.cloud/login.html --max-time 10 2>/dev/null || echo "henuz hazir degil"'))

# Traefik logları
print("\n=== Traefik loglar (yeni) ===")
print(ssh("docker logs coolify-proxy --tail 20 2>&1 | grep -i 'yadoba\\|cert\\|acme\\|obtained\\|error' | tail -10"))

client.close()
