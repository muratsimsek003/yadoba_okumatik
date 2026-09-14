# Depoya Ekleme Talimatı

`muratsimsek003/yadoba_okumatik` deposuna eklenecek değişiklikler.
Tüm testler geçmiş durumda (10/10).

## Dosyalar

| Dosya | Durum | Ne yapıyor |
|---|---|---|
| `ablation/run_ablation.py` | **YENİ** | Dört hizalama yöntemini karşılaştıran deney |
| `ablation/README.md` | **YENİ** | Deney tasarımı, sonuçlar, yorum, sınırlılık |
| `ablation/results/ablation_results.json` | **YENİ** | Ölçüm çıktısı (tohum sabit, yeniden üretilebilir) |
| `okumetrik/phonetics.py` | **DEĞİŞTİ** | Ayırt edici fonetik özellik tabanlı mesafe |
| `okumetrik/miscue.py` | **DEĞİŞTİ** | Sabit 0,34 eşiği kaldırıldı, modele bağlandı |
| `tests/test_engine.py` | **DEĞİŞTİ** | 4 yeni test eklendi (6 → 10) |

## Yöntem 1 — Dosyaları doğrudan kopyalayarak

```bash
cd /yol/yadoba_okumatik
git checkout -b feature/phoneme-feature-distance

# bu klasördeki dosyaları depo üzerine kopyalayın
cp -r github_ekleme/ablation      .
cp    github_ekleme/okumetrik/*.py okumetrik/
cp    github_ekleme/tests/*.py     tests/

python -m pytest tests/test_engine.py -q     # 10 passed beklenir
python ablation/run_ablation.py              # sonuçları yeniden üretir
```

## Yöntem 2 — Yamayı uygulayarak

`degisiklikler.patch`, yalnız üç mevcut dosyadaki değişiklikleri içerir
(`ablation/` klasörü yamada yoktur, ayrıca kopyalanmalıdır).

```bash
cd /yol/yadoba_okumatik
git checkout -b feature/phoneme-feature-distance
git apply github_ekleme/degisiklikler.patch
cp -r github_ekleme/ablation .
python -m pytest tests/test_engine.py -q
```

## Commit

```bash
git add ablation okumetrik/phonetics.py okumetrik/miscue.py tests/test_engine.py
git commit -m "Ayırt edici fonetik özellik tabanlı fonem mesafesi ve hizalama ablasyonu

- phonetics.py: çıkış yeri / çıkış biçimi / ötümlülük eksenlerinde fonem
  özellik tabloları; DISTANCE_MODE ile model seçimi (feature | class);
  maliyet ağırlıkları WEIGHTS sözlüğünde öğrenilebilir biçimde toplandı
- miscue.py: sabit MISPRON_THRESHOLD = 0.34 kaldırıldı, aktif mesafe
  modeline bağlandı (feature: 0.20, class: 0.34)
- mispronunciation_detail(): changed_features alanı eklendi, fonem farkı
  ayırt edici özellik düzeyinde açıklanıyor ('b -> p (ötümlülük)')
- ablation/: dört yöntemin (classic / ortho / class / feature) kontrollü
  1032 hata olayı üzerinde karşılaştırılması, eşik taraması dahil
- tests: 4 yeni test (6 -> 10)

Zorlayıcı koşulda Macro-F1: classic 0.4681, ortho 0.9361, class 0.9381,
feature 0.9479 (her yöntem kendi en iyi eşiğinde)."

git push -u origin feature/phoneme-feature-distance
```

## Neden bu değişiklik yapıldı

Kısa gerekçe (ayrıntısı `ablation/README.md` içinde):

1. Klasik sabit maliyetli hizalama, telaffuz sapması sınıfında F1 = 0,000
   veriyor — bu ayrımı yapısal olarak üretemiyor. Alt-sözcük benzerliği
   eklendiğinde Macro-F1 0,468 → 0,948. Projenin çekirdek mimari kararı doğru.
2. Ancak alt-sözcük yöntemleri arasındaki fark küçük: yazımsal 0,9361,
   sınıf temelli 0,9381, özellik temelli 0,9479. Türkçenin sığ ortografisi
   nedeniyle harf mesafesi zaten fonem mesafesine yaklaşıyor. Dolayısıyla
   "fonem mesafesi kullanmak" tek başına savunulabilir bir yenilik iddiası değil.
3. Eşik, mesafe modelinden daha belirleyici: sınıf temelli model eşiğe göre
   0,9381 ile 0,8108 arasında değişiyor. Kodda sabitlenmiş olan 0,34 değeri
   her iki fonem modeli için de en uygun değer değildi (class: 0,25,
   feature: 0,20). Eşiklerin uzman etiketli veriden öğrenilmesi gerekiyor.
4. Özellik temelli model, ayrışmayı asıl zor koşulda (fonetik komşu sözcük
   değiştirme) sağlıyor: class 0,9110'a düşerken feature 0,9479'da kalıyor.

## Sınırlılık — depoda ve başvuruda aynı şekilde belirtilmeli

Deney sentetiktir. Telaffuz sapmaları harf düzeyinde bozularak üretildiği için
yazımsal mesafe kayrılmaktadır; gerçek çocuk okumasındaki akustik sapmaların bir
kısmının yazımsal karşılığı yoktur. Sonuç nihai başarım kanıtı değil, yöntem
seçimini yönlendiren ön bulgudur.
