# Diyabet Riski Tahmini

Bu eğitim projesi, Pima Indians Diabetes veri setindeki klinik ölçümler üzerinden
farklı sınıflandırma algoritmalarını karşılaştırır. Çalışma; veri temizleme,
ölçeklendirme, model karşılaştırma ve hiperparametre arama adımlarını içerir.

> Bu proje tıbbi teşhis aracı değildir. Model çıktıları klinik karar veya tedavi
> amacıyla kullanılamaz.

## Uygulanan İş Akışı

1. Glikoz, kan basıncı ve BMI gibi alanlardaki tıbben geçersiz sıfırların incelenmesi
2. Doldurma değerlerinin yalnızca eğitim verisinden öğrenilmesi
3. Sınıf oranını koruyan eğitim-test ayrımı
4. Sayısal özelliklerin standartlaştırılması
5. Beş sınıflandırma algoritmasının karşılaştırılması
6. GridSearchCV ile hiperparametre araması

## Kayıtlı Sonuçlar

| Model | Test doğruluğu |
| --- | ---: |
| Naive Bayes | %75,97 |
| Lojistik Regresyon | %75,32 |
| Karar Ağacı | %74,68 |
| SVC | %74,68 |
| K-En Yakın Komşu | %71,43 |

Notebook'taki çapraz doğrulamada en yüksek kayıtlı skor yaklaşık **%76,9** ile
Lojistik Regresyon modeline aittir. Sağlık problemlerinde accuracy tek başına
yeterli olmadığından precision, recall, F1 ve karmaşıklık matrisi de incelenir.

## Proje Yapısı

```text
diabetes-risk-prediction-ml/
├── data/README.md
├── diabetes_risk_prediction.ipynb
├── requirements.txt
└── README.md
```

Veri lisansı doğrulanmadan veri dosyası depoya eklenmemiştir. Veri setini
`data/diabetes.csv` yoluna yerleştirin.

## Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook diabetes_risk_prediction.ipynb
```
