# Explainable Machine Learning for Heart Disease Risk Prediction: Comparative Evaluation of ML Algorithms, Feature Importance Stability and Clinical Interpretability

Explainable AI (XAI) okvir za predikciju rizika od srčanih bolesti koji modele
ne ocjenjuje samo prema prediktivnim performansama, nego i prema **stabilnosti**
i **kliničkoj smislenosti** njihovih objašnjenja (SHAP, LIME, PFI).

## Pregled

Implementirano je i upoređeno šest ML modela (Logistic Regression, Random
Forest, XGBoost, LightGBM, CatBoost, TabNet) na **dva strukturno različita
dataseta**, uz hiperparametarsku optimizaciju (Optuna) najboljih modela. Modeli
su dalje analizirani kroz više XAI metoda (SHAP, LIME, Permutation Feature
Importance), provjerena je **stabilnost objašnjenja** kroz različite seed-ove i
train/test podjele, te je izmjerena **klinička usklađenost** najvažnijih
karakteristika sa poznatim faktorima rizika iz medicinske literature
(ACC/AHA, Framingham Risk Score, ESC Guidelines).

### Dataseti

| | Dataset A | Dataset B |
|---|---|---|
| **Naziv** | Indicators of Heart Disease (CDC 2022) | Heart Failure Prediction |
| **Izvor** | CDC BRFSS Survey (Kaggle: `kamilpytlak/personal-key-indicators-of-heart-disease`) | Cleveland + Hungarian + Switzerland + VA (Kaggle: `fedesoriano/heart-failure-prediction`) |
| **Veličina** | ~440.000 zapisa | 918 pacijenata |
| **Tip** | Anketni / populacijski podaci | Klinički podaci |
| **Target** | `HadHeartAttack` | `HeartDisease` |
| **Uloga** | Glavni dataset — puna evaluacija | Validacijski — provjera robusnosti metodologije |

Isti pipeline, isti algoritmi i ista XAI metodologija primjenjuju se na oba
dataseta nezavisno, a poređenje SHAP rangova karakteristika između dataseta je
jedan od ključnih doprinosa rada.

## Struktura projekta

```
.
├── S3_Preprocessing_EDA.ipynb   # Učitavanje, čišćenje, EDA, enkodiranje, scaling, SMOTE
├── S4_Baseline_Models.ipynb     # Logistic Regression, Random Forest
├── S5_Main_Models.ipynb         # XGBoost, LightGBM, CatBoost, TabNet + Optuna tuning
├── S6_XAI.ipynb                 # SHAP, LIME, PFI, stabilnost, klinička interpretabilnost
├── data/processed/              # Pickle sa pripremljenim train/test podacima (generisano)
├── models/                       # Sačuvani modeli i CSV sa rezultatima
├── figures/                       # Generisane vizualizacije (EDA, metrike, XAI)
├── 2020/, 2022/                  # CDC dataset (preuzima se preko Kaggle API-ja)
└── heart.csv                      # Heart Failure dataset
```

## Pipeline

1. **S3 — Preprocessing & EDA**
   - Provjera kvaliteta podataka (missing values, duplikati), čišćenje
   - EDA: distribucije, korelacije, balans klasa, kategoričke vs. target
   - Enkodiranje (Ordinal/Label), standardizacija, train/test split
   - SMOTE **isključivo na trening skupu** (sprečava data leakage)
   - Rezultat: `data/processed/preprocessing_output.pkl`

2. **S4 — Baseline modeli**
   - Logistic Regression (`solver='saga'`, `C=1.0`)
   - Random Forest (`n_estimators=100`, `max_features='sqrt'`)
   - Evaluacija: Accuracy, Balanced Accuracy, Precision/Recall/F1
     (macro, weighted, klasa HD), Specificity, ROC-AUC, PR-AUC

3. **S5 — Glavni modeli**
   - XGBoost, LightGBM, CatBoost (gradient boosting)
   - TabNet (deep learning, attention-based, tabelarni podaci)
   - Hyperparameter tuning top-2 modela putem Bayesijanske optimizacije
     (Optuna, 50 trials, na stratificiranom poduzorku od 100k radi brzine)
   - Finalna usporedna tabela svih modela

4. **S6 — Explainable AI (XAI)**
   - **SHAP**: globalna (bar/beeswarm) i lokalna (waterfall) analiza, dependence plotovi
   - **LIME**: lokalna objašnjenja za TP/FN/TN/FP primjere
   - **Permutation Feature Importance (PFI)**: model-agnostička globalna metoda, poređenje sa SHAP (Jaccard, Spearman)
   - **Stabilnost objašnjenja**: kroz 5 random seed-ova i 5 nezavisnih 80/20 train/test podjela (cijeli pipeline ponovljen)
   - **Klinička interpretabilnost**: poređenje top-K SHAP karakteristika sa medicinskim faktorima rizika
   - **Cross-dataset usporedba**: SHAP rangiranje Dataset A vs Dataset B
   - **Composite score**: `0.40 × ROC-AUC + 0.20 × F1(macro) + 0.20 × Stabilnost(Jaccard) + 0.20 × Klinička(%)`

## Rezultati (sažetak)

| Model | Dataset | Accuracy | Balanced Acc. | F1 (macro) | ROC-AUC | PR-AUC | Recall (HD) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | A | 0.822 | 0.795 | 0.612 | 0.878 | 0.391 | 0.765 |
| Random Forest | A | 0.940 | 0.704 | 0.710 | 0.879 | 0.380 | 0.438 |
| XGBoost | A | 0.947 | 0.618 | 0.659 | 0.887 | 0.411 | 0.247 |
| LightGBM | A | 0.947 | 0.613 | 0.653 | 0.886 | 0.411 | 0.237 |
| CatBoost | A | 0.948 | 0.616 | 0.658 | 0.886 | 0.420 | 0.241 |
| TabNet | A | 0.818 | 0.780 | 0.605 | 0.862 | 0.347 | 0.736 |
| XGBoost (tuned) | A | 0.940 | 0.720 | 0.718 | 0.881 | 0.395 | 0.472 |
| CatBoost (tuned) | A | 0.939 | 0.719 | 0.718 | 0.880 | 0.405 | 0.471 |
| Logistic Regression | B | 0.870 | 0.862 | 0.866 | 0.879 | 0.875 | 0.931 |
| Random Forest | B | 0.902 | 0.900 | 0.901 | 0.927 | 0.914 | 0.922 |
| XGBoost | B | 0.859 | 0.859 | 0.858 | 0.926 | 0.934 | 0.853 |
| LightGBM | B | 0.875 | 0.875 | 0.874 | 0.932 | 0.936 | 0.873 |
| CatBoost | B | 0.875 | 0.877 | 0.874 | 0.931 | 0.935 | 0.863 |
| TabNet | B | 0.870 | 0.862 | 0.866 | 0.933 | 0.944 | 0.931 |
| XGBoost (tuned) | B | 0.880 | 0.875 | 0.878 | 0.921 | 0.917 | 0.922 |
| CatBoost (tuned) | B | 0.897 | 0.891 | 0.895 | 0.931 | 0.931 | 0.941 |

> Detaljne metrike, confusion matrice, ROC/PR krive i sva XAI poređenja
> dostupna su u `models/all_results.csv`, `models/baseline_results.csv` i
> direktorijima `figures/` i `figures/xai/`.

## Pokretanje projekta

### 1. Instalacija zavisnosti

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn ^
            xgboost lightgbm catboost pytorch-tabnet torch optuna shap lime ^
            python-dotenv kaggle
```

### 2. Kaggle API token

Dataseti se preuzimaju automatski preko Kaggle API-ja. Kreirajte `.env` fajl u
root direktoriju projekta sa sadržajem:

```
KAGGLE_API_TOKEN=<vaš_kaggle_api_token>
```

### 3. Redoslijed izvršavanja notebook-a

Notebook-i se izvršavaju **po redoslijedu**, jer svaki sljedeći koristi
podatke/modele generisane u prethodnom:

1. `S3_Preprocessing_EDA.ipynb` → preuzima dataseta, radi EDA i preprocessing,
   čuva `data/processed/preprocessing_output.pkl`
2. `S4_Baseline_Models.ipynb` → trenira baseline modele, čuva
   `models/baseline_models.pkl` i `models/baseline_results.csv`
3. `S5_Main_Models.ipynb` → trenira napredne modele i tuning, čuva
   `models/main_models.pkl` i `models/all_results.csv`
4. `S6_XAI.ipynb` → SHAP/LIME/PFI analiza, stabilnost, klinička
   interpretabilnost, čuva `models/xai_results.pkl`
