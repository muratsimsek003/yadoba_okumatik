#!/usr/bin/env python3
"""
OKUMETRiK Web Uygulamasi - FastAPI Backend
Kurulum: pip install fastapi uvicorn python-multipart
Calistir: python app.py
"""
import os, sys, json, uuid, hashlib, sqlite3
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "okumetrik.db"

# ── Whisper (lazy-load) ────────────────────────────────────────────────────
_model = None
_processor = None
MODEL_PATH = str(BASE_DIR / "whisper-child-tr-poc")

def load_whisper():
    global _model, _processor
    if _model is not None:
        return True
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else "openai/whisper-large-v3"
        _processor = WhisperProcessor.from_pretrained(path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
        print(f"[Whisper] {path} @ {device}")
        return True
    except Exception as e:
        print(f"[Whisper] Yuklenemedi: {e}")
        return False

# ── SQLite ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'student',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS texts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            level INTEGER DEFAULT 1,
            word_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            text_id TEXT NOT NULL,
            transcript TEXT,
            accuracy REAL DEFAULT 0,
            wcpm REAL DEFAULT 0,
            fluency_score REAL DEFAULT 0,
            duration_sec REAL DEFAULT 0,
            analysis_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    def h(pw): return hashlib.sha256(pw.encode()).hexdigest()

    for row in [
        (str(uuid.uuid4()), 'admin',    h('admin123'),    'Yonetici',       'admin'),
        (str(uuid.uuid4()), 'ogretmen', h('ogretmen123'), 'Ogretmen Ahmet', 'admin'),
        (str(uuid.uuid4()), 'ogrenci1', h('ogrenci123'),  'Ahmet Yilmaz',   'student'),
        (str(uuid.uuid4()), 'ogrenci2', h('ogrenci456'),  'Ayse Kaya',      'student'),
        (str(uuid.uuid4()), 'ogrenci3', h('ogrenci789'),  'Mehmet Demir',   'student'),
    ]:
        conn.execute(
            "INSERT OR IGNORE INTO users (id,username,password_hash,full_name,role) VALUES (?,?,?,?,?)", row)

    texts = [
        ("Kucuk Prens", "Bir zamanlar kucuk bir adam vardi. Bu adamin kucuk bir gezegeni vardi. Gezegende tek bir gul cicegi yetisiyordu. Adam bu gulu cok seviyordu. Her sabah suluyor her aksam ortuyordu. Gunler gectikce adam daha da yalniz hissediyordu. Sonunda gulu birakip baska gezegenlere yolculuga cikti.", 1),
        ("Heidinin Daglari", "Heidi kucuk bir kiz cocuguydu. Buyukbabasiyla dagda yasiyordu. Sabahlari erken kalkip koyunlari otlatmaya gidiyordu. Daglarin havasi temizdi ve cicekler her yerde aciyordu. Heidi bu guzellikleri cok seviyordu. Arkadasi Klara onu ziyarete gelince birlikte oyun oynadilar.", 1),
        ("Kaplumbaga ve Tavsan", "Bir ormanda kaplumbaga ile tavsan yarisimaya karar verdiler. Tavsan cok hizliydi ve kaplumbagayla yarisimak ona komik geldi. Kosimaya basladilar. Tavsan hemen ileriye gecti ve kaplumbaga cok geride kaldi. Tavsan yorulunca bir agacin altinda uyudu. Kaplumbaga ise durmadan yavas yavas yurudu. Tavsan uyanadiginda kaplumbaga coktan bitisi gecmisti.", 2),
        ("Pinokyo'nun Maceralari", "Marangoz Geppetto ahsaptan bir kukla yapti. Kuklanin adini Pinokyo koydu. Bir gun Pinokyo canlandi ve konusmaya basladi. Cocuga donusmek icin okula gitmeliydi. Ama Pinokyo yolda oyun parkina gitti. Yalanlar soyleyince burnu uzadi. Sonunda iyi bir cocuk olmaya karar verdi ve ruyasi gercek oldu.", 2),
        ("Ormanin Sirri", "Ormanin derinliklerinde kucuk bir koy vardi. Bu koyden yasayan cocuklar her gun ormana gidip oynarlardi. Bir gun kucuk Ali ormanda parlayan bir tas buldu. Tasi aldiginda etrafinda garip sesler duymaya basladi. Agaclar sallaniyor kuslar sarki soyleyordu. Ali tasi dikkatle inceledi ve uzerinde eski harfler gordu. Koye dondugunude herkes ona inanmadi ama Ali biliyordu ki ormanin bir sirri vardi.", 3),
        ("Denizkizi Masali", "Denizin dibinde guzel bir saray vardi. Bu sarayda yasayan denizkizi insanlarin dunyasini merak ediyordu. Her gun dalgalarin uzerine cikip uzaktaki gemileri izliyordu. Bir gun buyuk bir firtina cikti ve denizde bir gemi batti. Denizkizi bogulmakta olan bir prens gordu. Onu kurtarip kiyiya birakti ve denize geri dondu. Prensi hic unutamadi.", 3),
    ]
    for title, content, level in texts:
        conn.execute(
            "INSERT OR IGNORE INTO texts (id,title,content,level,word_count) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), title, content, level, len(content.split())))

    conn.commit()
    conn.close()
    print("[DB] Hazirlandi:", DB_PATH)


# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="OKUMETRiK", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup(): init_db()

# HTML sayfalar
for _page in ["login", "reading", "admin", "index"]:
    _path = f"/{_page}.html"
    _file = str(BASE_DIR / f"{_page}.html")
    app.add_api_route(_path, lambda p=_file: FileResponse(p), methods=["GET"])

@app.get("/")
def root(): return FileResponse(str(BASE_DIR / "login.html"))

# ── Auth ───────────────────────────────────────────────────────────────────
def current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Giris gerekli")
    token = authorization[7:]
    conn = get_db()
    row = conn.execute(
        "SELECT u.* FROM tokens t JOIN users u ON t.user_id=u.id WHERE t.token=?", (token,)
    ).fetchone()
    conn.close()
    if not row: raise HTTPException(401, "Gecersiz token")
    return dict(row)

def require_admin(user=Depends(current_user)):
    if user['role'] != 'admin': raise HTTPException(403, "Yonetici yetkisi gerekli")
    return user

class LoginReq(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
def login(req: LoginReq):
    ph = hashlib.sha256(req.password.encode()).hexdigest()
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?", (req.username, ph)
    ).fetchone()
    if not user:
        conn.close(); raise HTTPException(401, "Hatali kullanici adi veya sifre")
    token = str(uuid.uuid4())
    conn.execute("INSERT INTO tokens (token,user_id) VALUES (?,?)", (token, user['id']))
    conn.commit(); conn.close()
    return {"token": token, "user": {
        "id": user['id'], "username": user['username'],
        "full_name": user['full_name'], "role": user['role']
    }}

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        conn = get_db()
        conn.execute("DELETE FROM tokens WHERE token=?", (authorization[7:],))
        conn.commit(); conn.close()
    return {"ok": True}

@app.get("/api/me")
def me(user=Depends(current_user)):
    return {"id": user['id'], "username": user['username'],
            "full_name": user['full_name'], "role": user['role']}

# ── Texts ──────────────────────────────────────────────────────────────────
@app.get("/api/texts")
def list_texts(user=Depends(current_user)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM texts ORDER BY level,title").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/texts/{tid}")
def get_text(tid: str, user=Depends(current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM texts WHERE id=?", (tid,)).fetchone()
    conn.close()
    if not row: raise HTTPException(404, "Metin bulunamadi")
    return dict(row)

class TextReq(BaseModel):
    title: str
    content: str
    level: int = 1

@app.post("/api/texts")
def create_text(req: TextReq, user=Depends(require_admin)):
    tid = str(uuid.uuid4())
    wc = len(req.content.split())
    conn = get_db()
    conn.execute("INSERT INTO texts (id,title,content,level,word_count) VALUES (?,?,?,?,?)",
                 (tid, req.title, req.content, req.level, wc))
    conn.commit(); conn.close()
    return {"id": tid, "title": req.title, "content": req.content, "level": req.level, "word_count": wc}

@app.put("/api/texts/{tid}")
def update_text(tid: str, req: TextReq, user=Depends(require_admin)):
    wc = len(req.content.split())
    conn = get_db()
    conn.execute("UPDATE texts SET title=?,content=?,level=?,word_count=? WHERE id=?",
                 (req.title, req.content, req.level, wc, tid))
    conn.commit(); conn.close()
    return {"ok": True}

@app.delete("/api/texts/{tid}")
def delete_text(tid: str, user=Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM texts WHERE id=?", (tid,))
    conn.commit(); conn.close()
    return {"ok": True}

# ── Transcribe ─────────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), user=Depends(current_user)):
    import tempfile
    audio_bytes = await audio.read()
    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes); tmp = f.name

    try:
        import librosa
        waveform, _ = librosa.load(tmp, sr=16000)
        duration = len(waveform) / 16000

        if load_whisper():
            import torch
            inputs = _processor(waveform, sampling_rate=16000, return_tensors="pt")
            device = next(_model.parameters()).device
            feats = inputs.input_features.to(device)
            with torch.no_grad():
                ids = _model.generate(feats, language="tr", task="transcribe",
                                      no_repeat_ngram_size=0)
            transcript = _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        else:
            transcript = "[Demo mod — Whisper yuklenmedi]"
    finally:
        os.unlink(tmp)

    return {"transcript": transcript, "duration_sec": round(duration, 2)}

# ── Analyze ────────────────────────────────────────────────────────────────
class AnalyzeReq(BaseModel):
    reference: str
    hypothesis: str
    duration_sec: Optional[float] = None

@app.post("/api/analyze")
def analyze_api(req: AnalyzeReq, user=Depends(current_user)):
    sys.path.insert(0, str(BASE_DIR))
    from okumetrik import analyze
    report, entries = analyze(req.reference, req.hypothesis, audio_duration_sec=req.duration_sec)
    tokens = [{"status": e.status, "ref": e.ref, "hyp": e.hyp,
               "ref_idx": e.ref_idx, "hyp_idx": e.hyp_idx} for e in entries]
    return {"success": True, "report": {
        "metrics": report["metrics"],
        "overall_feedback": report["overall_feedback"],
        "word_feedback": report["word_feedback"],
        "tokens": tokens
    }}

# ── Sessions ───────────────────────────────────────────────────────────────
class SessionReq(BaseModel):
    text_id: str
    transcript: str
    accuracy: float
    wcpm: float
    fluency_score: float
    duration_sec: float
    analysis_json: str

@app.post("/api/sessions")
def create_session(req: SessionReq, user=Depends(current_user)):
    sid = str(uuid.uuid4())
    conn = get_db()
    conn.execute("""INSERT INTO sessions
        (id,user_id,text_id,transcript,accuracy,wcpm,fluency_score,duration_sec,analysis_json)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (sid, user['id'], req.text_id, req.transcript,
         req.accuracy, req.wcpm, req.fluency_score, req.duration_sec, req.analysis_json))
    conn.commit(); conn.close()
    return {"id": sid}

@app.get("/api/sessions")
def list_sessions(user=Depends(current_user)):
    conn = get_db()
    if user['role'] == 'admin':
        rows = conn.execute("""SELECT s.*,u.full_name user_name,t.title text_title
            FROM sessions s JOIN users u ON s.user_id=u.id JOIN texts t ON s.text_id=t.id
            ORDER BY s.created_at DESC""").fetchall()
    else:
        rows = conn.execute("""SELECT s.*,u.full_name user_name,t.title text_title
            FROM sessions s JOIN users u ON s.user_id=u.id JOIN texts t ON s.text_id=t.id
            WHERE s.user_id=? ORDER BY s.created_at DESC""", (user['id'],)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/sessions/{sid}")
def get_session(sid: str, user=Depends(current_user)):
    conn = get_db()
    row = conn.execute("""SELECT s.*,u.full_name user_name,t.title text_title,t.content text_content
        FROM sessions s JOIN users u ON s.user_id=u.id JOIN texts t ON s.text_id=t.id
        WHERE s.id=?""", (sid,)).fetchone()
    conn.close()
    if not row: raise HTTPException(404, "Oturum bulunamadi")
    r = dict(row)
    if r.get('analysis_json'):
        r['analysis'] = json.loads(r['analysis_json'])
    return r

# ── Users (admin) ──────────────────────────────────────────────────────────
@app.get("/api/users")
def list_users(user=Depends(require_admin)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id,username,full_name,role,created_at FROM users ORDER BY role,full_name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

class UserReq(BaseModel):
    username: str
    password: str
    full_name: str
    role: str = 'student'

@app.post("/api/users")
def create_user(req: UserReq, user=Depends(require_admin)):
    uid = str(uuid.uuid4())
    ph = hashlib.sha256(req.password.encode()).hexdigest()
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (id,username,password_hash,full_name,role) VALUES (?,?,?,?,?)",
                     (uid, req.username, ph, req.full_name, req.role))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Bu kullanici adi zaten kullanimda")
    finally:
        conn.close()
    return {"id": uid, "username": req.username, "full_name": req.full_name, "role": req.role}

@app.put("/api/users/{uid}")
def update_user_password(uid: str, body: dict, user=Depends(require_admin)):
    if 'password' not in body: raise HTTPException(400, "password gerekli")
    ph = hashlib.sha256(body['password'].encode()).hexdigest()
    conn = get_db()
    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (ph, uid))
    conn.commit(); conn.close()
    return {"ok": True}

@app.delete("/api/users/{uid}")
def delete_user(uid: str, user=Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit(); conn.close()
    return {"ok": True}

# ── Stats ──────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats(user=Depends(require_admin)):
    conn = get_db()
    stats = {
        "total_students": conn.execute("SELECT COUNT(*) FROM users WHERE role='student'").fetchone()[0],
        "total_texts":    conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0],
        "total_sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        "avg_accuracy":   round(
            conn.execute("SELECT COALESCE(AVG(accuracy),0) FROM sessions").fetchone()[0], 1),
        "recent_sessions": [dict(r) for r in conn.execute("""
            SELECT s.id,u.full_name user_name,t.title text_title,
                   s.accuracy,s.wcpm,s.fluency_score,s.created_at
            FROM sessions s
            JOIN users u ON s.user_id=u.id
            JOIN texts t ON s.text_id=t.id
            ORDER BY s.created_at DESC LIMIT 10""").fetchall()]
    }
    conn.close()
    return stats

# ── Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
