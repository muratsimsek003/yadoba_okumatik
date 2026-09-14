# Hizalama Yöntemi Ablasyon Deneyi

Bu klasör, TÜBİTAK 1501 başvurusunun (öneri no 197686) B.3 "Teknik ön fizibilite
çalışmaları" bölümüne konan hizalama bulgusunu üretir.

## Amaç

Projenin merkezî yöntem iddiasını — sözcük hizalama maliyetinin alt-sözcük
benzerliğine bağlanmasını — üç baz çizgiye karşı sınamak.

| # | Yöntem | Açıklama |
|---|---|---|
| 1 | `classic` | Klasik sabit maliyetli sözcük hizalama (yerine koyma maliyeti 1,0) |
| 2 | `ortho` | Yazımsal (karakter Levenshtein) mesafe — literatürdeki rakip yaklaşım |
| 3 | `class` | Ünlü/ünsüz sınıfı temelli fonem mesafesi (`okumetrik` ilk sürüm) |
| 4 | `feature` | Ayırt edici fonetik özellik temelli fonem mesafesi (`okumetrik` güncel) |

## Çalıştırma

Depo kökünden:

```bash
python ablation/run_ablation.py            # dört yöntem, iki koşul
python ablation/run_ablation.py --sweep    # + eşik taraması (yavaş, ~40 dk)
```

Rastgele tohum sabittir (`20260828`); sonuçlar yeniden üretilebilir.
Çıktı: `ablation/results/ablation_results.json`

## Veri

`pre_feasibility/synth_school_data.py` içindeki ilkokul 1-4. sınıf düzeyi 65
cümleden türetilen 257 sözcüklük dağarcık. Hata türü ve konumu bilinen 1.032
kontrollü hata olayı üretilir:

- **Telaffuz sapması** — tek fonem düzeyinde bozulma (ünlü değişimi, ünsüz
  ötümlülük değişimi, son/iç ses düşmesi). Üretilen biçim sözlükte bulunmaz.
- **Sözcük değiştirme** — gerçek sözcükle değiştirme. *Zorlayıcı koşulda* hedefe
  fonetik olarak en yakın 5 gerçek sözcük arasından seçilir.
- **Atlama** — sözcüğün hipotezden çıkarılması.

## Sonuçlar

### Ortak eşikte (0,34) — ilk ölçüm

| Yöntem | Kolay koşul | Zorlayıcı koşul |
|---|---|---|
| classic | 0,4639 | 0,4681 |
| ortho (0,34) | 0,9383 | 0,9361 |
| class (0,34) | 0,9507 | 0,9110 |
| feature (0,20) | 0,9411 | **0,9479** |

### Her yöntem kendi en iyi eşiğinde (zorlayıcı koşul)

| Yöntem | En iyi eşik | Macro-F1 |
|---|---|---|
| classic | — | 0,4681 |
| ortho | 0,34 | 0,9361 |
| class | 0,25 | 0,9381 |
| feature | 0,20 | **0,9479** |

### Eşik duyarlılığı (zorlayıcı koşul)

| Eşik | ortho | class | feature |
|---|---|---|---|
| 0,15 | — | — | 0,9184 |
| 0,20 | 0,8628 | — | **0,9479** |
| 0,25 | 0,8994 | **0,9381** | 0,9443 |
| 0,30 | 0,8959 | 0,9281 | 0,9064 |
| 0,34 | **0,9361** | 0,9110 | — |
| 0,35 | — | — | 0,8493 |
| 0,40 | 0,9171 | 0,8108 | — |

## Yorum

**1. Asıl kazanç alt-sözcük benzerliğinden gelir, ve büyüktür.**
Klasik sabit maliyetli hizalama telaffuz sapması sınıfında F1 = 0,000 verir;
bu ayrımı kötü yapmaz, *yapısal olarak hiç üretemez*. Sözcükleri bölünmez birim
saydığı için "kırmızı → kırmız" ile "balık → kuş" onun için aynı işlemdir.
Alt-sözcük benzerliği eklendiğinde Macro-F1 0,468'den 0,948'e çıkar (+0,48).
Projenin çekirdek mimari kararı bu bulguyla desteklenmektedir.

**2. Alt-sözcük yöntemleri arasındaki fark küçüktür.**
En iyi eşiklerinde ortho 0,9361, class 0,9381, feature 0,9479. Ayırt edici
özellik temelli model en iyisidir, ancak yazımsal baz çizgiye üstünlüğü bu
sentetik korpusta yalnız +0,012'dir. Türkçenin sığ ortografisi nedeniyle harf
mesafesi zaten fonem mesafesine yaklaşmaktadır. **Bu nedenle "fonem mesafesi
kullanmak" tek başına bir yenilik iddiası olarak savunulamaz.**

**3. Eşik, mesafe modelinden daha belirleyicidir.**
`class` modeli eşiğe göre 0,9381 ile 0,8108 arasında değişmektedir — yani yanlış
eşik seçimi, mesafe modelini değiştirmekten çok daha fazla başarım kaybettirir.
Kodda başlangıçta sabitlenmiş olan 0,34 değeri her iki fonem modeli için de
en uygun değer değildir (`class` için 0,25, `feature` için 0,20). Bu bulgu,
karar eşiklerinin sezgisel olarak atanamayacağını, uzman etiketli veriden
öğrenilmesi gerektiğini göstermektedir ve projedeki kalibrasyon iş paketinin
(İP3) doğrudan gerekçesidir.

**4. Özellik temelli model, zor koşulda ayrışır.**
Kolay koşulda (rastgele sözcük değiştirme) `class` modeli `feature`'dan biraz
daha iyidir; zorlayıcı koşulda (fonetik komşu sözcük değiştirme) ise `class`
0,9110'a düşerken `feature` 0,9479'da kalmaktadır. Pedagojik olarak asıl önemli
durum zorlayıcı koşuldur: çocuk hedef sözcüğe benzeyen başka bir sözcük
okuduğunda, telaffuz sapmasından ayırmak gerekir.

## Sınırlılık

Deney **sentetiktir**. Telaffuz sapmaları harf düzeyinde bozularak üretilmiştir;
bu üretim biçimi tanımı gereği yazımsal mesafeyi kayırır, çünkü bozulmalar zaten
harf uzayında tanımlıdır. Gerçek çocuk okumasında sapmalar akustik uzayda oluşur
(ünlü merkezileşmesi, kısmi sesletim, uzatma, akıcı olmayan geçişler) ve bir
kısmının yazımsal karşılığı yoktur. 257 sözcüklük dağarcık da gerçek okuma
metinlerinden küçüktür; dağarcık büyüdükçe fonetik olarak yakın gerçek sözcük
sayısı artar ve ayrım zorlaşır.

Bu nedenle sonuç **nihai başarım kanıtı değil, yöntem seçimini yönlendiren ön
bulgudur**. Gerçek çocuk konuşmasındaki başarım, uzman etiketli veriyle kalibre
edilerek ve konuşmacı-ayrık kilitli test kümesinde bağımsız olarak ölçülecektir.

## Koda etkisi

Bu deneyin sonucunda `okumetrik/phonetics.py` güncellenmiştir:

- Ayırt edici fonetik özellik tabloları (`VOWEL_FEATURES`, `CONSONANT_FEATURES`)
  eklendi; çıkış yeri / çıkış biçimi / ötümlülük eksenleri modellendi.
- Mesafe modeli `DISTANCE_MODE` ile seçilebilir hâle geldi (`"feature"` varsayılan,
  `"class"` baz çizgi olarak korundu).
- Maliyet ağırlıkları `WEIGHTS` sözlüğünde açık ve öğrenilebilir biçimde toplandı.
- Karar eşiği modele bağlandı (`mispron_threshold()`); `miscue.py` içindeki sabit
  0,34 değeri kaldırıldı.
- `mispronunciation_detail()` artık `changed_features` alanı döndürüyor:
  `"b → p (ötümlülük)"` gibi, öğretmene sunulabilir açıklama.
