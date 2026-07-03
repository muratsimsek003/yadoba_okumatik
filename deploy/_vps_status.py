import paramiko, os
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=20):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8","replace").strip()
    err = stderr.read().decode("utf-8","replace").strip()
    return (out + err).strip()

print("=== DNS ===")
print("yadoba @8.8.8.8 :", ssh("dig +short yadoba.isteknoloji.cloud @8.8.8.8"))
print("yadoba @1.1.1.1 :", ssh("dig +short yadoba.isteknoloji.cloud @1.1.1.1"))
print("NS kayitlari    :", ssh("dig +short NS isteknoloji.cloud @8.8.8.8"))

print("\n=== Servis ===")
print("yadoba aktif    :", ssh("systemctl is-active yadoba"))
print("port 3001       :", ssh("ss -tlnp | grep 3001 | head -1"))
print("HTTP test       :", ssh('curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3001/login.html'))

print("\n=== Traefik yadoba config ===")
print(ssh("cat /data/coolify/proxy/dynamic/yadoba.yaml"))

print("\n=== Traefik router listesi ===")
print(ssh('curl -s http://localhost:8080/api/http/routers 2>/dev/null | python3 -c "import sys,json; rs=json.load(sys.stdin); [print(r[\'name\'],\'|\',r.get(\'rule\',\'\'),\'|\',r.get(\'status\',\'\')) for r in rs if \'yadoba\' in r.get(\'name\',\'\').lower()]" 2>/dev/null || echo "API erisim yok"'))

print("\n=== Traefik son loglar ===")
print(ssh("docker logs coolify-proxy --tail 20 2>&1 | grep -i 'yadoba\\|cert\\|acme\\|error' | tail -10"))

client.close()
