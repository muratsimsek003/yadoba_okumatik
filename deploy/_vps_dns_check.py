import paramiko, os
os.environ["PYTHONIOENCODING"] = "utf-8"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("72.61.189.209", username="root", password=".6XgS?zDlh?hPC7f", timeout=15)

def ssh(cmd, timeout=15):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8","replace").strip()

print("=== DNS Kontrolu ===")
print("Google DNS (8.8.8.8):")
print(ssh("dig +short yadoba.isteknoloji.cloud @8.8.8.8"))
print("Cloudflare DNS (1.1.1.1):")
print(ssh("dig +short yadoba.isteknoloji.cloud @1.1.1.1"))
print("kok domain (isteknoloji.cloud) NS kayitlari:")
print(ssh("dig +short NS isteknoloji.cloud @8.8.8.8"))
print("kok domain A kaydi:")
print(ssh("dig +short isteknoloji.cloud @8.8.8.8"))

client.close()
