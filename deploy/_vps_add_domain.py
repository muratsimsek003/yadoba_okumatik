import paramiko, os
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8","replace").strip()
    err = stderr.read().decode("utf-8","replace").strip()
    if out: print(out)
    if err: print("ERR:", err[:300])
    return out

DOMAIN = "yadoba.isteknoloji.cloud"

# Traefik dynamic config dosyası
# host.docker.internal → VPS'in host IP'si (Traefik container içinden erişim)
config = f"""http:
  routers:
    yadoba-http:
      entryPoints:
        - http
      rule: Host(`{DOMAIN}`)
      middlewares:
        - yadoba-https-redirect
      service: yadoba
      priority: 10

    yadoba-https:
      entryPoints:
        - https
      rule: Host(`{DOMAIN}`)
      service: yadoba
      tls:
        certResolver: letsencrypt
      priority: 10

  middlewares:
    yadoba-https-redirect:
      redirectScheme:
        scheme: https
        permanent: true

  services:
    yadoba:
      loadBalancer:
        servers:
          - url: http://host.docker.internal:3001
        passHostHeader: true
"""

print(f"=== Traefik config yaziliyor: {DOMAIN} ===")
# Güvenli write: sftp ile
sftp = client.open_sftp()
with sftp.open("/data/coolify/proxy/dynamic/yadoba.yaml", "w") as f:
    f.write(config)
sftp.close()
print("Dosya yazildi.")

print("\n=== Config içeriği ===")
ssh("cat /data/coolify/proxy/dynamic/yadoba.yaml")

print("\n=== Traefik config reload (otomatik - file watch aktif) ===")
# Traefik dosyayı otomatik algılar ama zorlamak için:
ssh("docker kill --signal=SIGHUP coolify-proxy 2>/dev/null && echo RELOAD_OK || echo 'zaten izliyor'")

print("\n=== DNS propagasyon kontrolü ===")
out = ssh(f"dig +short {DOMAIN} @8.8.8.8 2>/dev/null || nslookup {DOMAIN} 8.8.8.8 2>/dev/null | grep Address | tail -1")
if "72.61.189.209" in out:
    print(f"DNS OK -> {out}")
else:
    print(f"DNS henuz yayilmamis (normal, 1-5 dk bekleyin): {out}")

print(f"\n=== Tamamlandi ===")
print(f"1. DNS kaydini Hostinger'da ekleyin (yadoba A 72.61.189.209)")
print(f"2. 1-5 dakika bekleyin (DNS yayilmasi)")
print(f"3. https://{DOMAIN} adresini tarayicida acin")
print(f"   SSL sertifikasi ilk acilista otomatik alinacak.")

client.close()
