# Diyabet Riski Tahmini

Klinik ölçümlerden diyabet riskini sınıflandırmak amacıyla farklı makine öğrenmesi
algoritmalarını karşılaştıran uçtan uca bir veri bilimi çalışması.

> **Proje türü:** Sınıflandırma · **Odak:** Sağlık verisi · **En yüksek kayıtlı test doğruluğu:** %75,97

## Projenin Amacı

Erken risk değerlendirmesinde yalnızca model doğruluğu değil, riskli bireylerin ne
ölçüde yakalanabildiği de önemlidir. Bu proje; Pima Indians Diabetes veri setindeki
klinik değişkenleri kullanarak farklı sınıflandırma yaklaşımlarının performansını
karşılaştırır ve sağlık verilerinde doğru değerlendirme metriği seçiminin önemini
gösterir.

Bu çalışma eğitim amaçlıdır; tıbbi teşhis veya tedavi aracı değildir.

## Veri ve Ön İşleme

Veri setinde gebelik sayısı, glikoz, kan basıncı, deri kalınlığı, insülin, BMI, yaş
ve aile geçmişini temsil eden değişkenler yer alır. Klinik olarak anlamlı olmayan
sıfır değerleri incelenmiş ve doldurma değerleri yalnızca eğitim verisinden
öğrenilerek veri sızıntısı önlenmiştir.

Uygulanan süreç:

1. Eksik ve tıbben geçersiz değerlerin belirlenmesi
2. Sınıf dağılımını koruyan eğitim-test ayrımı
3. Sayısal özelliklerin standartlaştırılması
4. Beş farklı sınıflandırma algoritmasının karşılaştırılması
5. Çapraz doğrulama ve GridSearchCV ile model seçimi
6. Accuracy, precision, recall, F1 ve karmaşıklık matrisiyle değerlendirme

## Sonuçlar

| Model | Test doğruluğu |
| --- | ---: |
| Naive Bayes | **%75,97** |
| Lojistik Regresyon | %75,32 |
| Karar Ağacı | %74,68 |
| SVC | %74,68 |
| K-En Yakın Komşu | %71,43 |

Notebook’taki çapraz doğrulamada en yüksek kayıtlı skor yaklaşık **%76,9** ile
Lojistik Regresyon modeline aittir. Modellerin birbirine yakın sonuç vermesi,
algoritma seçiminden önce veri kalitesi ve özelliklerin açıklayıcılığının
geliştirilmesi gerektiğine işaret etmektedir.

Sağlık problemlerinde yanlış negatiflerin maliyeti yüksek olabileceği için başarı
yalnızca accuracy ile yorumlanmamalıdır. Recall, precision ve karar eşiği sonraki
çalışmalarda kullanım senaryosuna göre ayrıca ele alınmalıdır.

## Proje Yapısı

```text
diabetes-risk-prediction-ml/
├── data/
│   └── README.md
├── diabetes_risk_prediction.ipynb
├── requirements.txt
└── README.md
```

Veri lisansı doğrulanmadan veri dosyası depoya eklenmemiştir. Veri setini
`data/diabetes.csv` konumuna yerleştirin.

## Kurulum ve Çalıştırma

```bash
git clone https://github.com/sametcsk/diabetes-risk-prediction-ml.git
cd diabetes-risk-prediction-ml
python -m venv .venv
pip install -r requirements.txt
jupyter notebook diabetes_risk_prediction.ipynb
```

## Kullanılan Teknolojiler

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter`

## Sınırlılıklar ve Sonraki Adımlar

- Sonuçlar tek bir veri setine dayandığı için farklı popülasyonlara doğrudan genellenemez.
- ROC-AUC ve PR-AUC metrikleriyle daha kapsamlı karşılaştırma yapılabilir.
- Özellik önemleri ve SHAP değerleriyle model kararları açıklanabilir.
- Karar eşiği, yanlış negatif maliyetine göre optimize edilebilir.
