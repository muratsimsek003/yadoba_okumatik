#!/usr/bin/env python3
"""
YADOBA Web Uygulamasi - FastAPI Backend
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
_fw_model = None  # faster-whisper
MODEL_PATH = str(BASE_DIR / "whisper-child-tr-poc")
# faster-whisper kullanır (large-v3, int8 — ~2GB RAM, transformers large-v3 kalitesi)
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")

def load_whisper():
    global _fw_model, _model, _processor
    if _fw_model is not None:
        return "faster"
    # faster-whisper tercih edilir
    try:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else WHISPER_MODEL_NAME
        _fw_model = WhisperModel(path, device=device, compute_type=compute_type)
        print(f"[faster-whisper] {path} @ {device} ({compute_type})")
        return "faster"
    except Exception as e:
        print(f"[faster-whisper] Yuklenemedi: {e}, transformers'a geciliyor...")
    # transformers fallback
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        hf_path = (MODEL_PATH if os.path.exists(MODEL_PATH)
                   else f"openai/whisper-{WHISPER_MODEL_NAME}" if not WHISPER_MODEL_NAME.startswith("openai/")
                   else WHISPER_MODEL_NAME)
        _processor = WhisperProcessor.from_pretrained(hf_path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = WhisperForConditionalGeneration.from_pretrained(hf_path, torch_dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
        print(f"[transformers-whisper] {hf_path} @ {device}")
        return "transformers"
    except Exception as e:
        print(f"[Whisper] Yuklenemedi: {e}")
        return None

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
        (str(uuid.uuid4()), 'admin',    h('admin123'),    'Yönetici',       'admin'),
        (str(uuid.uuid4()), 'ogretmen', h('ogretmen123'), 'Öğretmen Ahmet', 'admin'),
    ]:
        conn.execute(
            "INSERT OR IGNORE INTO users (id,username,password_hash,full_name,role) VALUES (?,?,?,?,?)", row)

    texts = [
        ("Küçük Prens",
         "Bir zamanlar küçük bir adam vardı. Bu adamın küçük bir gezegeni vardı. "
         "Gezegende tek bir gül çiçeği yetişiyordu. Adam bu gülü çok seviyordu. "
         "Her sabah sulardı, her akşam üstünü örterdi. Günler geçtikçe adam daha da "
         "yalnız hissediyordu. Sonunda gülü bırakıp başka gezegenlere yolculuğa çıktı.", 1),

        ("Heidi'nin Dağları",
         "Heidi küçük bir kız çocuğuydu. Büyükbabasıyla dağda yaşıyordu. "
         "Sabahları erken kalkıp koyunları otlatmaya gidiyordu. Dağların havası "
         "temizdi ve çiçekler her yerde açıyordu. Heidi bu güzellikleri çok seviyordu. "
         "Arkadaşı Klara onu ziyarete gelince birlikte oyun oynadılar.", 1),

        ("Çirkin Ördek Yavrusu",
         "Bir göl kenarında ördek yumurtaları çatlayıp açıldı. Bütün yavrular sarı "
         "ve güzeldi ama biri farklıydı. O büyük ve gri görünüyordu. Diğer hayvanlar "
         "onunla alay ettiler. Üzgün yavru yalnız başına yaşamaya başladı. Kış "
         "geçti, ilkbahar geldi. Yavru suya bakınca gördü ki o artık güzel bir "
         "kuğuya dönüşmüştü.", 1),

        ("Kaplumbağa ve Tavşan",
         "Bir ormanda kaplumbağa ile tavşan yarışmaya karar verdiler. Tavşan çok "
         "hızlıydı ve kaplumbağayla yarışmak ona komik geldi. Koşmaya başladılar. "
         "Tavşan hemen ileriye geçti ve kaplumbağa çok geride kaldı. Tavşan "
         "yorulunca bir ağacın altında uyudu. Kaplumbağa ise durmadan yavaş yavaş "
         "yürüdü. Tavşan uyandığında kaplumbağa çoktan bitişi geçmişti.", 2),

        ("Ağustos Böceği ve Karınca",
         "Yaz boyunca ağustos böceği şarkı söyleyip dans etti. Karınca ise durmadan "
         "çalışarak kışa hazırlandı. Sonbahar gelince ağustos böceği karıncadan yiyecek "
         "istedi. Karınca dedi ki yaz boyunca ne yaptın? Ağustos böceği utandı. "
         "Bu masaldan şunu öğreniriz çalışmak her zaman gereklidir.", 2),

        ("Pinokyo'nun Maceraları",
         "Marangoz Geppetto ahşaptan bir kukla yaptı. Kuklanın adını Pinokyo koydu. "
         "Bir gün Pinokyo canlandı ve konuşmaya başladı. Çocuğa dönüşmek için okula "
         "gitmeliydi. Ama Pinokyo yolda oyun parkına gitti. Yalanlar söyleyince burnu "
         "uzadı. Sonunda iyi bir çocuk olmaya karar verdi ve rüyası gerçek oldu.", 2),

        ("Ormanın Sırrı",
         "Ormanın derinliklerinde küçük bir köy vardı. Bu köyde yaşayan çocuklar her "
         "gün ormana gidip oynarlardı. Bir gün küçük Ali ormanda parlayan bir taş "
         "buldu. Taşı aldığında etrafında garip sesler duymaya başladı. Ağaçlar "
         "sallanıyor, kuşlar şarkı söylüyordu. Ali taşı dikkatle inceledi ve üzerinde "
         "eski harfler gördü. Köye döndüğünde herkes ona inanmadı ama Ali biliyordu "
         "ki ormanın bir sırrı vardı.", 3),

        ("Deniz Kızı Masalı",
         "Denizin dibinde güzel bir saray vardı. Bu sarayda yaşayan deniz kızı "
         "insanların dünyasını merak ediyordu. Her gün dalgaların üzerine çıkıp "
         "uzaktaki gemileri izliyordu. Bir gün büyük bir fırtına çıktı ve denizde "
         "bir gemi battı. Deniz kızı boğulmakta olan bir prens gördü. Onu kurtarıp "
         "kıyıya bıraktı ve denize geri döndü. Prensi hiç unutamadı.", 3),

        ("Mevsimlerin Dansı",
         "İlkbaharda çiçekler açar, ağaçlar yeşillenir. Çocuklar bahçede koşup oynar. "
         "Yazın güneş erken doğar ve geç batar. Herkes denize, göle, ormana gider. "
         "Sonbaharda yapraklar sarıya kızıla döner ve rüzgarda uçuşur. "
         "Kışın kar yağar, her yer bembeyaz olur. Çocuklar kardan adam yapar, "
         "karda oynar. Dört mevsim birbirini izler, doğa hiç durmadan değişir.", 1),

        ("İlk Uçuş",
         "Kuşlar yuvada büyür ve bir gün uçmayı öğrenmek zorunda kalırlar. "
         "Küçük serçe yuvadan baktı, aşağısı çok derin görünüyordu. Annesi yanında "
         "durdu ve kanatlarını çırpmayı gösterdi. Serçe derin bir nefes aldı ve "
         "yuvadan atladı. Kanatları titredi, biraz sendeledi ama uçuyordu. "
         "Hava onu taşıyordu. Korkusu sevince dönüşmüştü.", 2),

        ("Güneş Sistemi",
         "Güneş sisteminde sekiz gezegen vardır. En büyüğü Jüpiter, en küçüğü Merkür'dür. "
         "Dünya, Güneş'e üçüncü en yakın gezegendir. Dünya'nın bir uydusu vardır, adı Ay'dır. "
         "Mars'a kırmızı gezegen denir çünkü toprağı kırmızımsı demir oksitten oluşur. "
         "Satürn'ün çevresinde buz ve kayadan oluşan halkalar vardır. "
         "Gezegenler milyonlarca yıldır Güneş'in etrafında dönmeye devam etmektedir.", 3),

        ("Arkadaşlık",
         "Arkadaşlık hayatın en güzel hediyelerinden biridir. İyi bir arkadaş zor "
         "günlerde yanınızda olur. Sevincinizi paylaşır, üzüntünüzde sizi teselli eder. "
         "Arkadaşlık vermek ve almak demektir. Sırlarınızı saklayan, sizi olduğunuz gibi "
         "kabul eden insanlar gerçek arkadaşlardır. Bir arkadaşa iyi davranmak "
         "ona verdiğiniz en büyük hediyedir.", 2),
    ]
    for title, content, level in texts:
        wc = len(content.split())
        # Aynı başlıkta metin varsa ekleme (sunucu yeniden başladığında tekrar eklenmesini önler)
        exists = conn.execute("SELECT id FROM texts WHERE title=?", (title,)).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO texts (id,title,content,level,word_count) VALUES (?,?,?,?,?)",
                (str(uuid.uuid4()), title, content, level, wc))

    conn.commit()
    conn.close()
    print("[DB] Hazirlandi:", DB_PATH)


# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Depends, File, Form, UploadFile, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="YADOBA", version="2.0")
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

class RegisterReq(BaseModel):
    username: str
    password: str
    full_name: str

@app.post("/api/auth/register")
def register(req: RegisterReq):
    if len(req.username.strip()) < 3:
        raise HTTPException(400, "Kullanıcı adı en az 3 karakter olmalı")
    if len(req.password) < 6:
        raise HTTPException(400, "Şifre en az 6 karakter olmalı")
    if not req.full_name.strip():
        raise HTTPException(400, "Ad Soyad boş olamaz")
    ph = hashlib.sha256(req.password.encode()).hexdigest()
    uid = str(uuid.uuid4())
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (id,username,password_hash,full_name,role) VALUES (?,?,?,?,?)",
            (uid, req.username.strip().lower(), ph, req.full_name.strip(), 'student')
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(409, "Bu kullanıcı adı zaten alınmış, başka bir tane deneyin")
    token = str(uuid.uuid4())
    conn.execute("INSERT INTO tokens (token,user_id) VALUES (?,?)", (token, uid))
    conn.commit(); conn.close()
    return {"token": token, "user": {
        "id": uid, "username": req.username.strip().lower(),
        "full_name": req.full_name.strip(), "role": "student"
    }}

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

class StoryReq(BaseModel):
    title: str = "Kendi Hikayem"
    content: str

@app.post("/api/story/start")
def start_story(req: StoryReq, user=Depends(current_user)):
    if not req.content.strip():
        raise HTTPException(400, "Hikaye boş olamaz")
    title = req.title.strip() or "Kendi Hikayem"
    tid = str(uuid.uuid4())
    wc = len(req.content.split())
    conn = get_db()
    conn.execute("INSERT INTO texts (id,title,content,level,word_count) VALUES (?,?,?,?,?)",
                 (tid, title, req.content.strip(), 0, wc))
    conn.commit(); conn.close()
    return {"id": tid, "title": title, "content": req.content.strip(), "level": 0, "word_count": wc}

@app.delete("/api/texts/{tid}")
def delete_text(tid: str, user=Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM texts WHERE id=?", (tid,))
    conn.commit(); conn.close()
    return {"ok": True}

# ── Transcribe ─────────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    text_id: Optional[str] = Form(None),
    user=Depends(current_user)
):
    import tempfile, subprocess
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Ses verisi boş — kayıt düzgün çalışmadı")

    # Referans metni initial_prompt olarak kullan (Türkçe karakter doğruluğunu artırır)
    initial_prompt = None
    if text_id:
        conn = get_db()
        text_row = conn.execute("SELECT content FROM texts WHERE id=?", (text_id,)).fetchone()
        conn.close()
        if text_row:
            initial_prompt = text_row['content'][:200]  # İlk 200 karakter yeterli

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1]

    tmp = None
    wav_tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp = f.name

        # ffmpeg ile webm → wav dönüşümü (en güvenilir yol)
        wav_tmp = tmp.replace(suffix, "_converted.wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp, "-ar", "16000", "-ac", "1", "-f", "wav", wav_tmp],
            capture_output=True, timeout=60
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-300:]
            # ffmpeg başarısız olduysa librosa ile dene
            try:
                import librosa
                waveform, _ = librosa.load(tmp, sr=16000)
            except Exception as e2:
                raise HTTPException(500, f"Ses dönüştürülemedi. ffmpeg: {err} | librosa: {e2}")
        else:
            import librosa
            waveform, _ = librosa.load(wav_tmp, sr=16000)

        duration = float(len(waveform)) / 16000.0

        backend = load_whisper()
        if backend == "faster":
            segments, _ = _fw_model.transcribe(
                wav_tmp or tmp, language="tr", task="transcribe",
                initial_prompt=initial_prompt,
                temperature=0.0,   # Greedy decoding — en tutarlı çıktı
                beam_size=5,
                vad_filter=True,   # Sessiz kısımları filtrele
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
        elif backend == "transformers":
            import torch
            inputs = _processor(waveform, sampling_rate=16000, return_tensors="pt")
            device = next(_model.parameters()).device
            dtype  = next(_model.parameters()).dtype
            feats  = inputs.input_features.to(device=device, dtype=dtype)
            with torch.no_grad():
                ids = _model.generate(feats, language="tr", task="transcribe",
                                      no_repeat_ngram_size=0)
            transcript = _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        else:
            transcript = "[Demo mod — Whisper yuklenmedi]"

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transkript hatası: {e}")
    finally:
        for f in [tmp, wav_tmp]:
            if f:
                try: os.unlink(f)
                except: pass

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

# ── Resimden Hikaye (3-adımlı: llava denetle+tanı, qwen2 Türkçe yaz) ────────
@app.post("/api/story/from-image")
async def story_from_image(
    image: UploadFile = File(...),
    story: str = Form(...),
    user=Depends(current_user)
):
    import base64, asyncio
    import urllib.request as _req
    import urllib.error as _uerr

    if not story.strip():
        raise HTTPException(400, "Hikaye boş olamaz")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Resim yüklenemedi")

    image_b64 = base64.standard_b64encode(image_bytes).decode()

    ollama_url        = os.environ.get("OLLAMA_URL",         "http://localhost:11434")
    vision_model      = os.environ.get("OLLAMA_VISION_MODEL", "llava")
    text_model        = os.environ.get("OLLAMA_TEXT_MODEL",   "aya:8b")

    def _ollama(model, prompt, images=None, max_tokens=300):
        payload = {
            "model":      model,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": 0,          # RAM'den hemen boşalt
            "options":    {"num_predict": max_tokens, "temperature": 0.7},
        }
        if images:
            payload["images"] = images
        data = json.dumps(payload).encode()
        req  = _req.Request(
            f"{ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _req.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode()).get("response", "").strip()
        except _uerr.HTTPError as he:
            raise RuntimeError(f"Ollama HTTP {he.code}: {he.read().decode('utf-8','replace')[:200]}")
        except _uerr.URLError as ue:
            raise RuntimeError(f"Ollama bağlantı hatası: {ue.reason}")

    loop = asyncio.get_running_loop()

    try:
        # ── Adım 1: İçerik denetimi ──────────────────────────────────────────
        safety_prompt = (
            "You are a content moderator for a children's app.\n"
            "Look at this image. Does it contain violence, nudity, adult content, "
            "or anything inappropriate for children aged 6-12?\n"
            "Reply with ONE word only: SAFE or UNSAFE."
        )
        safety = await loop.run_in_executor(
            None, _ollama, vision_model, safety_prompt, [image_b64], 5
        )
        print(f"[story/from-image] safety check: {safety!r}", flush=True)

        if "UNSAFE" in safety.upper():
            raise HTTPException(400, "Bu görsel çocuklar için uygun değil. Lütfen farklı bir resim yükleyin.")

        # ── Adım 2: Resmi İngilizce tanımla (llava bu işi iyi yapıyor) ───────
        desc_prompt = (
            "Describe what you see in this image in one sentence. "
            "Mention: setting, main objects, colors, and mood."
        )
        description = await loop.run_in_executor(
            None, _ollama, vision_model, desc_prompt, [image_b64], 80
        )
        print(f"[story/from-image] image desc: {description!r}", flush=True)

        # ── Adım 3: qwen2 ile Türkçe hikaye ──────────────────────────────────
        story_prompt = (
            f"Resimde görülenler: {description}\n"
            f"Çocuğun yazdığı hikaye: {story.strip()}\n\n"
            "Yukarıdaki resim açıklamasını ve çocuğun hikayesini kullanarak "
            "kısa bir Türkçe çocuk hikayesi yaz.\n\n"
            "ZORUNLU KURALLAR:\n"
            "1. TAM OLARAK 2 PARAGRAF yaz — ne eksik ne fazla\n"
            "2. Her paragraf en fazla 3 kısa, basit cümle\n"
            "3. Resimdeki unsurları hikayeye yansıt\n"
            "4. Çocuklara uygun, sade Türkçe kullan\n"
            "5. SADECE hikayeyi yaz — başka açıklama, giriş veya yorum ekleme\n\n"
            "1. Paragraf:\n"
        )
        improved = await loop.run_in_executor(
            None, _ollama, text_model, story_prompt, None, 300
        )
        print(f"[story/from-image] story generated ({len(improved)} chars)", flush=True)

        # "1. Paragraf:" önekini modelin çıktısına ekledik, gerisi geldi
        improved = ("1. Paragraf:\n" + improved).strip()
        # Eğer model "1. Paragraf:" / "2. Paragraf:" etiketleri yazdıysa temizle
        import re
        improved = re.sub(r"^\d\.\s*Paragraf:\s*", "", improved, flags=re.MULTILINE).strip()

        if not improved:
            raise HTTPException(500, "LLM boş yanıt döndü")

    except HTTPException:
        raise
    except RuntimeError as e:
        print(f"[story/from-image] RuntimeError: {e}", flush=True)
        raise HTTPException(503, str(e))
    except Exception as e:
        import traceback
        print(f"[story/from-image] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}", flush=True)
        raise HTTPException(500, f"LLM hatası: {type(e).__name__}: {str(e)[:300]}")

    return {"improved": improved}

# ── Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
