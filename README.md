# Makine Öğrenmesi ile Diyabet Verisi Analizi

Bu eğitim projesi, Pima Indians Diabetes veri setinde farklı sınıflandırma
algoritmalarını karşılaştırır. Amaç klinik tanı koymak değil, veri ön işleme,
model değerlendirme ve hiperparametre arama adımlarını uygulamaktır.

## Uygulanan Adımlar

- Glikoz ve kan basıncı gibi alanlardaki geçersiz sıfır değerlerinin incelenmesi
- Eksik veya geçersiz değerlerin medyan ile doldurulması
- Sayısal özelliklerin standartlaştırılması
- Beş sınıflandırma algoritmasının karşılaştırılması
- GridSearchCV ile hiperparametre araması

## Temel Sonuçlar

| Model | Test doğruluğu |
| --- | ---: |
| Naive Bayes | %75,97 |
| Lojistik Regresyon | %75,32 |
| Karar Ağacı | %74,68 |
| SVC | %74,68 |
| K-En Yakın Komşu | %71,43 |

Karar ağacı eğitim verisinde %100 doğruluğa ulaştığı hâlde test doğruluğu
%74,68'de kalmıştır. Bu fark aşırı öğrenme riskine işaret eder; bu nedenle tek
başına en iyi model olarak değerlendirilmemelidir.

## Çalıştırma

```bash
git clone https://github.com/sametcsk/Makine-Ogrenmesi-ile-Diyabet-Analizi.git
cd Makine-Ogrenmesi-ile-Diyabet-Analizi
jupyter notebook diyabet-analizi.ipynb
```

> Bu çalışma yalnızca eğitim amaçlıdır; tıbbi teşhis veya tedavi amacıyla kullanılamaz.
