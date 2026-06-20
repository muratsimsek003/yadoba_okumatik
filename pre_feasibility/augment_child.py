"""
augment_child.py — Çocuk sesi simülasyonu için ses artırma araçları.

YADOBA domain'i: 6-12 yaş Türkçe çocuk sesli okuması.
FLEURS gibi yetişkin ses veri setlerini çocuk ses özelliklerine yaklaştırmak için
pitch shift + tempo yavaşlatma uygular.

Kullanım:
    from augment_child import augment_dataset
    ds_aug = augment_dataset(ds_fleurs, text_col="transcription")
"""
import io
import random
import numpy as np


def pitch_shift_child(audio: np.ndarray, sr: int = 16000,
                      semitones_range: tuple = (3, 7)) -> np.ndarray:
    """Sesi çocuk pitch aralığına taşı (+3 ile +7 yarı ton).

    Yetişkin temel frekansı ~85-255 Hz; çocuk ~250-500 Hz.
    Ortalama +5 yarı ton ≈ ×1.33 frekans — gerçekçi yaklaşım.
    """
    import librosa
    n_steps = random.uniform(*semitones_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def tempo_slow(audio: np.ndarray, sr: int = 16000,
               rate_range: tuple = (0.75, 0.92)) -> np.ndarray:
    """Okuma temposunu yavaşlat (pitch koruyarak).

    Çocuklar yetişkinlere göre %8-25 daha yavaş okur; heceleri uzatır.
    librosa.effects.time_stretch pitch'i korur, sadece hız değişir.
    """
    import librosa
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)


def add_room_noise(audio: np.ndarray, snr_db_range: tuple = (15, 30)) -> np.ndarray:
    """Hafif oda gürültüsü ekle (SNR 15-30 dB).

    Sınıf/ev ortamı simülasyonu. Gerçek çocuk kayıtlarında her zaman
    arka plan sesi vardır.
    """
    snr_db = random.uniform(*snr_db_range)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def augment_sample(audio: np.ndarray, sr: int = 16000,
                   do_pitch: bool = True,
                   do_tempo: bool = True,
                   do_noise: bool = True) -> np.ndarray:
    """Tek bir ses örneğine çocuk simülasyonu uygula."""
    if do_pitch:
        audio = pitch_shift_child(audio, sr)
    if do_tempo:
        audio = tempo_slow(audio, sr)
    if do_noise:
        audio = add_room_noise(audio)
    return audio.astype(np.float32)


def augment_dataset(ds, text_col: str = "transcription",
                    sr: int = 16000,
                    decode_fn=None,
                    augment_ratio: float = 1.0,
                    seed: int = 42):
    """HuggingFace dataset'e çocuk augmentasyonu uygula.

    Args:
        ds: datasets.Dataset — audio sütunu Audio(decode=False) ile yüklenmiş
        text_col: transkript sütun adı
        sr: hedef örnekleme hızı
        decode_fn: ham audio bytes → np.ndarray fonksiyonu (decode_audio)
        augment_ratio: veri setinin kaçta kaçına augment uygulanacak (1.0 = tümü)
        seed: tekrarlanabilirlik

    Returns:
        Augment edilmiş (audio_array, text) çiftlerinden oluşan liste.
        Her eleman: {"array": np.ndarray, "text": str}
    """
    import librosa

    if decode_fn is None:
        def decode_fn(item):
            raw = item.get("bytes") or item.get("path")
            if isinstance(raw, bytes):
                arr, _ = librosa.load(io.BytesIO(raw), sr=sr, mono=True)
            else:
                arr, _ = librosa.load(raw, sr=sr, mono=True)
            return arr.astype(np.float32)

    rng = random.Random(seed)
    results = []
    for ex in ds:
        if rng.random() > augment_ratio:
            continue
        try:
            audio = decode_fn(ex["audio"])
            aug = augment_sample(audio, sr)
            results.append({"array": aug, "text": ex[text_col]})
        except Exception:
            pass
    return results


def augment_dataset_hf(ds, text_col: str = "transcription",
                       sr: int = 16000, decode_fn=None,
                       augment_ratio: float = 1.0, seed: int = 42):
    """augment_dataset'i HuggingFace Dataset'e dönüştür."""
    from datasets import Dataset
    samples = augment_dataset(ds, text_col, sr, decode_fn, augment_ratio, seed)
    return Dataset.from_list([
        {"audio": {"array": s["array"], "sampling_rate": sr},
         "transcription": s["text"]}
        for s in samples
    ])


if __name__ == "__main__":
    import librosa
    print("augment_child.py — hızlı test")
    sr = 16000
    dummy = np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr).astype(np.float32)

    shifted = pitch_shift_child(dummy, sr, semitones_range=(4, 6))
    slowed = tempo_slow(dummy, sr, rate_range=(0.8, 0.85))
    noisy = add_room_noise(dummy, snr_db_range=(20, 25))
    full = augment_sample(dummy, sr)

    print(f"  Orijinal  : {len(dummy)/sr:.2f}s, max={dummy.max():.3f}")
    print(f"  Pitch+5st : {len(shifted)/sr:.2f}s")
    print(f"  Yavaş×0.8 : {len(slowed)/sr:.2f}s")
    print(f"  Gürültülü : {len(noisy)/sr:.2f}s")
    print(f"  Tam augment: {len(full)/sr:.2f}s")
    print("  OK")
