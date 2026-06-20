"""
synth_school_data.py — Türkçe ilkokul okuma metni + gTTS ile sentetik veri üretimi.

YADOBA domain'i: 6-12 yaş çocukların okuduğu ilkokul 1-4. sınıf seviyesi cümleler.
gTTS ile Türkçe TTS sesi üretilir; isteğe bağlı olarak augment_child ile
çocuk sesi simülasyonu eklenir.

Kullanım:
    python synth_school_data.py --out data/synth_school --augment --n 200
"""
import argparse
import io
import os
import random

# İlkokul 1-4. sınıf Türkçe okuma metinlerinden alınan cümleler.
# Kelime hazinesi kasıtlı olarak basit tutulmuştur.
SCHOOL_SENTENCES = [
    # 1. sınıf seviyesi
    "Ali oku.",
    "Bak, bir kedi var.",
    "Anne eve geldi.",
    "Top kırmızıdır.",
    "Bebek uyuyor.",
    "Güneş parlıyor.",
    "Kuş uçuyor.",
    "Su içiyorum.",
    "Araba geçti.",
    "Çiçekler güzeldir.",
    # 2. sınıf seviyesi
    "Bugün hava çok güzel.",
    "Köpek bahçede koşuyor.",
    "Kitabı masaya koy.",
    "Annem çorba pişirdi.",
    "Kardeşim benimle oynadı.",
    "Okula erken geldim.",
    "Kelebek çiçeğe kondu.",
    "Ağaçlar rüzgarda sallandı.",
    "Balıklar suda yüzüyor.",
    "Çocuklar parkta oynuyor.",
    "Bahçede elmalar var.",
    "Sepeti taşıdım.",
    "Kışın kar yağar.",
    "Bahar gelince çiçekler açar.",
    "Güneş doğudan yükselir.",
    # 3. sınıf seviyesi
    "Türkiye güzel bir ülkedir.",
    "Ankara Türkiye'nin başkentidir.",
    "Mavi gökyüzünde bulutlar yüzüyordu.",
    "Öğretmenimiz bize hikâye anlattı.",
    "Kütüphanede birçok kitap bulunur.",
    "Temizlik sağlığın yarısıdır.",
    "Hayvanları sevmek güzel bir davranıştır.",
    "Yağmur yağdığında gökkuşağı çıkar.",
    "İnsanlar birbirine yardım etmelidir.",
    "Sebzeler sağlığımız için çok önemlidir.",
    "Okumak insanı bilgili yapar.",
    "Dürüstlük en güzel erdemdir.",
    "Çalışkan öğrenci her zaman başarılı olur.",
    "Arkadaşlarımla birlikte top oynadık.",
    "Kışın eldiven ve atkı takıyoruz.",
    # 4. sınıf seviyesi
    "Ülkemizde dört mevsim yaşanmaktadır.",
    "Türkçe dilimiz çok zengin bir dildir.",
    "Tarihî eserleri korumak hepimizin görevidir.",
    "Anadolu binlerce yıllık medeniyetlere ev sahipliği yapmıştır.",
    "Denizlerimizde birçok balık türü yaşar.",
    "Ormanlar dünyamızın akciğerleridir.",
    "Geri dönüşüm çevreyi korumanın önemli yollarından biridir.",
    "Matematikte problem çözmek dikkat gerektirir.",
    "Harita okumayı öğrenmek coğrafya dersinin temelini oluşturur.",
    "Spor yapmak hem beden hem de ruh sağlığına yararlıdır.",
    "Kitap okuma alışkanlığı kazanmak çok değerlidir.",
    "Su kaynaklarını israf etmemeye dikkat etmeliyiz.",
    "Bilgisayar teknolojisi hayatımızı kolaylaştırmaktadır.",
    "Kuşlar göç ederek uzak ülkelere giderler.",
    "Cumhuriyet bayramını her yıl coşkuyla kutluyoruz.",
    # Okuma akıcılığı testi cümleleri (WCPM odaklı)
    "Ali dün sabah erkenden kalktı ve okula gitmek için hazırlandı.",
    "Annesi ona kahvaltı hazırladı; yumurta, peynir ve ekmek vardı.",
    "Yolda arkadaşı Veli ile karşılaştı ve birlikte yürüdüler.",
    "Sınıfa girdiklerinde öğretmen tahtaya bir şeyler yazıyordu.",
    "Ders zili çaldığında herkes sessizce oturdu ve kitaplarını açtı.",
    "Öğle tatilinde bahçeye çıktılar ve güneşin altında oynadılar.",
    "Akşam eve döndüğünde önce ellerini yıkadı, sonra ödevini yaptı.",
    "Uyumadan önce annesine o günkü maceralarını anlattı.",
    "Yaşlı bir adam yolda düşmüştü, hemen koşup yardım ettim.",
    "Kütüphaneye gidip resimli kitaplar ödünç aldım.",
]


def generate_audio_gtts(text: str, lang: str = "tr") -> bytes:
    """gTTS ile Türkçe TTS sesi üret; OGG baytı döndür."""
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def mp3_to_wav_array(mp3_bytes: bytes, sr: int = 16000):
    """MP3 baytını numpy array'e dönüştür."""
    import librosa
    arr, _ = librosa.load(io.BytesIO(mp3_bytes), sr=sr, mono=True)
    return arr.astype("float32")


def build_synth_dataset(sentences: list[str],
                        out_dir: str,
                        augment: bool = True,
                        sr: int = 16000,
                        seed: int = 42) -> list[dict]:
    """Cümle listesinden TTS + (isteğe bağlı augmentasyon) ile veri seti üret.

    Döndürür:
        [{"array": np.ndarray, "sampling_rate": int, "transcription": str}, ...]
    """
    import numpy as np
    if augment:
        from augment_child import augment_sample

    random.seed(seed)
    samples = []
    os.makedirs(out_dir, exist_ok=True)

    for i, sent in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {sent[:50]}...")
        try:
            mp3_bytes = generate_audio_gtts(sent)
            arr = mp3_to_wav_array(mp3_bytes, sr)

            if augment:
                arr = augment_sample(arr, sr,
                                     do_pitch=True, do_tempo=True, do_noise=True)

            wav_path = os.path.join(out_dir, f"synth_{i:04d}.npy")
            np.save(wav_path, arr)
            samples.append({
                "array": arr,
                "sampling_rate": sr,
                "transcription": sent,
                "wav_path": wav_path,
            })
        except Exception as e:
            print(f"    HATA: {e}")

    return samples


def samples_to_hf_dataset(samples: list[dict]):
    """Örnek listesini HuggingFace Dataset'e dönüştür."""
    from datasets import Dataset, Audio
    rows = [{"audio": {"array": s["array"], "sampling_rate": s["sampling_rate"]},
             "transcription": s["transcription"]}
            for s in samples]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser(description="Türkçe okul metni TTS veri seti üretici")
    ap.add_argument("--out", default="data/synth_school", help="Çıktı dizini")
    ap.add_argument("--n", type=int, default=0,
                    help="Kullanılacak cümle sayısı (0 = tümü)")
    ap.add_argument("--augment", action="store_true",
                    help="Çocuk sesi augmentasyonu uygula")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_hf", action="store_true",
                    help="HuggingFace Dataset olarak kaydet")
    args = ap.parse_args()

    sentences = SCHOOL_SENTENCES
    if args.n and args.n < len(sentences):
        random.seed(args.seed)
        sentences = random.sample(sentences, args.n)

    print(f"Sentetik veri üretiliyor: {len(sentences)} cümle | augment={args.augment}")
    samples = build_synth_dataset(sentences, args.out,
                                  augment=args.augment, seed=args.seed)
    print(f"\n✅ {len(samples)} örnek üretildi → {args.out}/")

    if args.save_hf:
        hf_path = os.path.join(args.out, "hf_dataset")
        ds = samples_to_hf_dataset(samples)
        ds.save_to_disk(hf_path)
        print(f"   HF Dataset kaydedildi: {hf_path}")

    return samples


if __name__ == "__main__":
    main()
