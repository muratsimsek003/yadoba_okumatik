"""
OKUMETRİK — Türkçe sesli okuma değerlendirme motoru (YADOBA 2.0 çekirdeği).

Bu paket, referans metin ile okuma (tanıma) çıktısı arasındaki yanlış okuma
(miscue) ve fonem-düzeyi telaffuz hatalarını tespit eder, okuma akıcılığı
metriklerini üretir ve açıklanabilir Türkçe geri bildirim verir.

Çalıştırılabilir/test edilebilir kısım: hizalama + miscue sınıflandırma +
skorlama + Türkçe G2P. Akustik model (HF-MTM) eğitimi için bkz. ../asr_scaffold.
"""

from .pipeline import analyze, evaluate
from .miscue import tokenize, align
from .g2p import g2p
from .phonetics import phonetic_distance

__all__ = ["analyze", "evaluate", "tokenize", "align", "g2p", "phonetic_distance"]
__version__ = "0.1.0-poc"
