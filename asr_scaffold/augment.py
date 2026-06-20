"""
asr_scaffold.augment — çocuk konuşması için kaynak-filtre temelli veri artırımı.

Yetişkin konuşma korpuslarını çocuk konuşmasına perceptual olarak yaklaştırmak
için Vocal Tract Length Perturbation (VTLP) + hız/perde sapması uygulanır
(Thienpondt & Demuynck, 2022 yaklaşımına dayanır). Bu, az etiketli çocuk
verisi probleminde domain mismatch'i kapatmayı hedefler (İP3–İP4).

GEREKSİNİM: numpy (+ üretimde torchaudio). Bu ortamda numpy ile çalışır.
"""
from __future__ import annotations
import numpy as np


def vtlp_warp_spectrum(mag: np.ndarray, alpha: float) -> np.ndarray:
    """
    Genlik spektrumuna (frekans ekseni) parçalı doğrusal VTLP warp uygular.
    alpha > 1 : ses tellerini 'kısaltır' (çocuk konuşmasına yaklaştırır).
    mag: (freq_bins, frames)
    """
    F = mag.shape[0]
    src = np.arange(F)
    warped = np.where(src < 0.8 * F, src * alpha,
                      0.8 * F * alpha + (src - 0.8 * F) *
                      ((F - 0.8 * F * alpha) / (F - 0.8 * F)))
    warped = np.clip(warped, 0, F - 1)
    out = np.zeros_like(mag)
    for f in range(F):
        lo = int(np.floor(warped[f])); hi = min(lo + 1, F - 1)
        frac = warped[f] - lo
        out[f] = (1 - frac) * mag[lo] + frac * mag[hi]
    return out


def child_like_augment(waveform: np.ndarray, sr: int = 16000,
                       alpha_range=(1.05, 1.25),
                       speed_range=(0.95, 1.10)) -> np.ndarray:
    """
    Bir dalga formunu çocuk konuşmasına benzeyecek şekilde rastgele dönüştürür.
    (STFT -> VTLP -> ISTFT + hız sapması). Üretimde torchaudio ile hızlandırılır.
    """
    alpha = np.random.uniform(*alpha_range)
    speed = np.random.uniform(*speed_range)
    # Basit STFT
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    frames = []
    for start in range(0, len(waveform) - n_fft, hop):
        frames.append(np.fft.rfft(waveform[start:start + n_fft] * win))
    if not frames:
        return waveform
    spec = np.stack(frames, axis=1)                 # (freq, frames) complex
    mag, phase = np.abs(spec), np.angle(spec)
    mag = vtlp_warp_spectrum(mag, alpha)
    spec = mag * np.exp(1j * phase)
    # ISTFT (overlap-add)
    out = np.zeros(len(waveform))
    for k in range(spec.shape[1]):
        seg = np.fft.irfft(spec[:, k], n=n_fft) * win
        s = k * hop
        out[s:s + n_fft] += seg
    # Hız sapması (basit yeniden örnekleme)
    idx = np.clip(np.arange(0, len(out), speed), 0, len(out) - 1).astype(int)
    return out[idx]


def augment_batch(waveforms, sr=16000):
    return [child_like_augment(w, sr) for w in waveforms]
