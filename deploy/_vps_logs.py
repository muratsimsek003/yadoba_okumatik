import paramiko, os
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=20):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8","replace").strip() + stderr.read().decode("utf-8","replace").strip()

print("=== Son uygulama logları ===")
print(ssh("journalctl -u yadoba --no-pager -n 50 --output=short"))

client.close()
