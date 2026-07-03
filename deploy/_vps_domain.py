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

print("=== Mevcut Traefik docker-compose ===")
ssh("cat /data/coolify/proxy/docker-compose.yml")

print("\n=== Mevcut dynamic config ===")
ssh("cat /data/coolify/proxy/dynamic/default_redirect_503.yaml")

client.close()
