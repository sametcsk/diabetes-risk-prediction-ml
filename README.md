# Diabetes Risk Prediction with Machine Learning
# Makine Öğrenmesi ile Diyabet Riski Tahmini

End-to-end classification study comparing multiple ML algorithms on the
Pima Indians Diabetes dataset.

Pima Indians Diabetes veri seti üzerinde farklı ML algoritmalarını karşılaştıran
uçtan uca sınıflandırma çalışması.

> **Task / Görev:** Binary classification · **Domain / Alan:** Healthcare / Sağlık · **Best test accuracy / En iyi test doğruluğu:** 75.97 %

---

## Objective / Amaç

Early diabetes screening benefits from models that not only maximise accuracy
but also minimise false negatives. This project benchmarks six classifiers,
tunes them with `GridSearchCV`, and highlights why evaluation metrics beyond
accuracy matter in healthcare applications.

Erken diyabet taramasında yalnızca doğruluk değil, riskli bireylerin ne ölçüde
yakalanabildiği de önemlidir. Bu proje altı sınıflandırıcıyı karşılaştırır,
`GridSearchCV` ile ayarlar ve sağlık verilerinde doğru metrik seçiminin önemini
gösterir.

*Educational project — not a clinical diagnostic tool.*
*Eğitim amaçlıdır; tıbbi teşhis veya tedavi aracı değildir.*

## Results / Sonuçlar

| Model | Test Accuracy |
| --- | ---: |
| Naive Bayes | **75.97 %** |
| Logistic Regression | 75.32 % |
| Decision Tree | 74.68 % |
| SVC | 74.68 % |
| KNN | 71.43 % |

Cross-validation scores up to **76.9 %** were observed for Logistic Regression.

Çapraz doğrulamada en yüksek skor **%76.9** ile Lojistik Regresyon'a aittir.

## Project Structure / Proje Yapısı

```text
diabetes-risk-prediction-ml/
├── main.py                  # CLI entry point — run full pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load, validate & impute data
│   ├── feature_engineering.py   # Train/test split, scaling
│   ├── model.py             # Model definitions, evaluation, tuning
│   └── visualization.py     # Correlation heatmap, box plots, bar chart
├── notebooks/
│   └── diabetes_risk_prediction.ipynb   # Original exploratory notebook
├── data/
│   └── README.md            # Instructions for obtaining the dataset
├── requirements.txt
└── README.md
```

## Quick Start / Hızlı Başlangıç

```bash
git clone https://github.com/sametcsk/diabetes-risk-prediction-ml.git
cd diabetes-risk-prediction-ml
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt

# Place the dataset at data/diabetes.csv / Veri setini data/diabetes.csv konumuna yerleştirin
python main.py                    # full pipeline with plots / grafiklerle tam pipeline
python main.py --skip-plots       # headless / CI mode / grafik olmadan
```

## Pipeline Overview / Pipeline Adımları

1. **Data loading & validation** — detect clinically-invalid zero values
   *Veri yükleme & doğrulama — klinik olarak geçersiz sıfır değerleri tespit*
2. **Imputation** — replace zeros with training-set median (no data leakage)
   *İmpütasyon — sıfırları eğitim seti medyanıyla doldurma (veri sızıntısı yok)*
3. **Feature scaling** — `StandardScaler` fitted on training data only
   *Özellik ölçekleme — sadece eğitim verisinden fit edilen StandardScaler*
4. **Baseline evaluation** — train six classifiers, compare accuracy
   *Temel değerlendirme — altı sınıflandırıcıyı eğit, doğruluğu karşılaştır*
5. **Hyperparameter tuning** — `GridSearchCV` with 5-fold stratified CV
   *Hiperparametre ayarı — 5 katlı sınıf dengeli çapraz doğrulama*
6. **Final comparison** — test-set accuracy of the best estimators
   *Son karşılaştırma — en iyi modellerin test seti doğruluğu*

## Tech Stack / Kullanılan Teknolojiler

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

## Limitations & Next Steps / Sınırlılıklar & Sonraki Adımlar

- Results are specific to this dataset; generalisability is untested.
  *Sonuçlar tek veri setine özeldir; genellenebilirlik test edilmemiştir.*
- ROC-AUC and PR-AUC would provide a more complete picture.
  *ROC-AUC ve PR-AUC ile daha kapsamlı karşılaştırma yapılabilir.*
- SHAP values could explain individual predictions.
  *SHAP değerleriyle model kararları açıklanabilir.*
- Decision threshold optimisation could reduce false-negative cost.
  *Karar eşiği, yanlış negatif maliyetine göre optimize edilebilir.*

## License / Lisans

This project is provided for educational purposes.
Bu proje eğitim amaçlıdır.
