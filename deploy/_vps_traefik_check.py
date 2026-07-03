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
    if err: print("ERR:", err[:200])

print("=== Traefik container config ===")
ssh("docker inspect coolify-proxy --format '{{json .Mounts}}' | python3 -c \"import sys,json; [print(m['Source'],'->',m['Destination']) for m in json.load(sys.stdin)]\"")

print("\n=== Traefik config dosyaları ===")
ssh("find /data/coolify/proxy -name '*.yml' -o -name '*.yaml' -o -name '*.toml' 2>/dev/null | head -20")

print("\n=== Traefik dynamic config dizini ===")
ssh("ls /data/coolify/proxy/ 2>/dev/null || ls /etc/traefik/ 2>/dev/null || echo 'bulunamadı'")

print("\n=== Coolify proxy volume ===")
ssh("docker volume ls | grep proxy")
ssh("docker volume inspect coolify_proxy_data 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d[0]['Mountpoint'])\" 2>/dev/null || echo 'volume adı farklı'")

client.close()
