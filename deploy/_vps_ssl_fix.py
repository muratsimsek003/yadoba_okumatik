import paramiko, os, json, time
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

# acme.json'dan yadoba girdisini temizle
print("=== acme.json okunuyor ===")
acme_raw = ssh("cat /data/coolify/proxy/acme.json")
try:
    acme = json.loads(acme_raw)
    le = acme.get("letsencrypt", {})
    certs = le.get("Certificates", [])
    print(f"Mevcut sertifikalar: {[c.get('domain',{}).get('main','?') for c in certs]}")

    # yadoba.isteknoloji.cloud sertifikasını kaldır
    new_certs = [c for c in certs if "yadoba.isteknoloji.cloud" not in str(c.get("domain", ""))]
    le["Certificates"] = new_certs
    acme["letsencrypt"] = le
    new_acme = json.dumps(acme)

    # SFTP ile geri yaz
    sftp = client.open_sftp()
    with sftp.open("/data/coolify/proxy/acme.json", "w") as f:
        f.write(new_acme)
    sftp.close()
    print("acme.json temizlendi.")
except Exception as e:
    print(f"acme.json parse hatasi: {e}")
    # Dosyayı sıfırla
    sftp = client.open_sftp()
    with sftp.open("/data/coolify/proxy/acme.json", "w") as f:
        f.write('{}')
    sftp.close()
    print("acme.json sifirlandiverdi.")

# Traefik yeniden başlat
print("\n=== Traefik yeniden baslatiliyor ===")
print(ssh("docker restart coolify-proxy", 30))

print("\n=== 20 saniye bekleniyor (cert alinmasi) ===")
time.sleep(20)

print("\n=== Traefik yeni loglar ===")
print(ssh("docker logs coolify-proxy --since 60s 2>&1 | tail -20"))

print("\n=== HTTPS testi ===")
result = ssh('curl -sk -o /dev/null -w "%{http_code}" https://yadoba.isteknoloji.cloud/login.html --max-time 15 2>/dev/null')
print(f"HTTPS HTTP kodu: {result}")

if "200" in result or "301" in result or "302" in result:
    print("\n HTTPS CALISIYOR!")
    print("Site: https://yadoba.isteknoloji.cloud")
else:
    print("\n Sertifika henuz alinmamis, 1-2 dakika daha bekleyin.")

client.close()
