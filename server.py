#!/usr/bin/env python3
"""
OKUMETRİK Web Demonstratör Sunucusu.
Sıfır bağımlılıkla (sadece standart kitaplık kullanarak) çalışır.
"""
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import okumetrik pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from okumetrik import analyze

class OkumetrikRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format%args))

    def do_GET(self):
        # Serve index.html
        if self.path in ("/", "/index.html"):
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
            if os.path.exists(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        if self.path == "/api/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                reference = data.get("reference", "")
                hypothesis = data.get("hypothesis", "")
                audio_duration = data.get("audio_duration_sec")
                
                if audio_duration is not None:
                    try:
                        audio_duration = float(audio_duration)
                    except ValueError:
                        audio_duration = None
                
                # If hypothesis is a string comma/space separated, we pass it as string.
                # If it's formatted like a list of dicts, we pass the parsed json list.
                # Let's clean hypothesis string if it is passed as a string or list
                if isinstance(hypothesis, str):
                    # Clean it to simple words list
                    pass

                report, entries = analyze(reference, hypothesis, audio_duration_sec=audio_duration)
                
                # Build json serializable tokens
                tokens_response = []
                for e in entries:
                    tokens_response.append({
                        "status": e.status,
                        "ref": e.ref,
                        "hyp": e.hyp,
                        "ref_idx": e.ref_idx,
                        "hyp_idx": e.hyp_idx,
                        "detail": {k: v for k, v in e.detail.items() if k != "phoneme_ops"}
                    })
                
                response_data = {
                    "success": True,
                    "report": {
                        "metrics": report["metrics"],
                        "overall_feedback": report["overall_feedback"],
                        "word_feedback": report["word_feedback"],
                        "tokens": tokens_response
                    }
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
                
            except Exception as e:
                response_data = {"success": False, "error": str(e)}
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

def run(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, OkumetrikRequestHandler)
    print(f"OKUMETRİK Sunucusu http://localhost:{port} portunda başlatıldı...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatılıyor...")
        httpd.server_close()

if __name__ == "__main__":
    # Ensure stdout/stderr is UTF-8 to prevent console crashes
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port)
