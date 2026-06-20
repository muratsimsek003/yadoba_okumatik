"""
okumetrik.g2p — Türkçe kural-tabanlı grapheme-to-phoneme (G2P).

Türkçe yazımı büyük ölçüde şeffaf (sığ ortografi) olduğundan, kelime
telaffuzu kurallarla yüksek doğrulukla türetilebilir. Bu modül, fonem-düzeyi
telaffuz hatası analizinde (MDD) referans/hipotez kelimelerin fonem dizilerini
karşılaştırmak için kullanılır.

Çıktı, basitleştirilmiş bir fonem alfabesidir (IPA'ya yakın ASCII):
  ünlüler: a e ı i o ö u ü   -> a E I i o 2 u y
  ünsüzler: b c ç d f g ğ h j k l m n p r s ş t v y z
            -> b dZ tS d f g (ğ özel) h Z k l m n p r s S t v j z

Notlar:
- 'ğ' (yumuşak g): ünlüden sonra genelde sesli uzatması / kayma; burada
  kendisinden önceki ünlüyü uzatan boş fonem ':' olarak modellenir.
- Bu, mükemmel bir telaffuz sözlüğü DEĞİLDİR; PoC ve kıyas için yeterli,
  kural-tabanlı bir yaklaşımdır. Üretimde sözlük + istisna listesiyle genişletilir.
"""

VOWELS = set("aeıioöuü")

_MAP = {
    "a": "a", "e": "E", "ı": "I", "i": "i", "o": "o", "ö": "2", "u": "u", "ü": "y",
    "b": "b", "c": "dZ", "ç": "tS", "d": "d", "f": "f", "g": "g",
    "h": "h", "j": "Z", "k": "k", "l": "l", "m": "m", "n": "n",
    "p": "p", "r": "r", "s": "s", "ş": "S", "t": "t", "v": "v",
    "y": "j", "z": "z",
}

# Sesli/sessiz benzeşmesi gibi basit ses olayları için yardımcı kümeler
PLOSIVE_VOICED = {"b", "d", "g", "dZ"}
PLOSIVE_VOICELESS = {"p", "t", "k", "tS"}


def normalize(word: str) -> str:
    """Türkçe'ye duyarlı küçük harfe çevirme ve ayıklama."""
    word = word.strip()
    # Türkçe büyük I/İ inceliği:
    word = word.replace("I", "ı").replace("İ", "i")
    word = word.lower()
    return "".join(ch for ch in word if ch.isalpha() or ch in "ğçşöüı")


def g2p(word: str) -> list[str]:
    """Bir kelimeyi fonem listesine çevirir."""
    w = normalize(word)
    phonemes: list[str] = []
    i = 0
    while i < len(w):
        ch = w[i]
        if ch == "ğ":
            # Yumuşak g: önceki ünlüyü uzat (':'), kendisi sessiz kabul edilir.
            if phonemes:
                phonemes.append(":")
            i += 1
            continue
        ph = _MAP.get(ch)
        if ph is None:
            i += 1
            continue
        phonemes.append(ph)
        i += 1
    return phonemes


def is_vowel_phoneme(p: str) -> bool:
    return p in {"a", "E", "I", "i", "o", "2", "u", "y"}
