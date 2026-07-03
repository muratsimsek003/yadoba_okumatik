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

print("=== Docker konteynerleri ===")
ssh("docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'")

print("\n=== Dinleyen portlar ===")
ssh("ss -tlnp | grep LISTEN")

print("\n=== Mevcut siteler (nginx) ===")
ssh("ls /etc/nginx/sites-enabled/ 2>/dev/null")

print("\n=== yadoba servis logu ===")
ssh("journalctl -u yadoba --no-pager -n 20")

client.close()
