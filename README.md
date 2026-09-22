# AGROSOIL AI

## Intelligent Soil Health Analysis & Fertilizer Recommendation System

AGROSOIL AI is a premium Streamlit product experience for **DataXcelerate 2026, PS22**. Part 3 adds a supervised Random Forest prototype for soil fertility classification on top of the transparent Part 2 data foundation.

### Part 1 scope

- Premium AgriTech dashboard with responsive glassmorphism styling
- Hero landing screen with placeholder KPI values
- Soil report, soil image and farmer experience entry points
- Soil intelligence feature overview
- Visual AI workflow from input to recommendation
- About page and polished placeholders for future modules
- No fake datasets, predictions, recommendations or API keys

### Part 2 completed

- Reusable Pandas data loader with required-column and data-quality validation
- Explicit **Demo / Synthetic Dataset** at `data/raw/soil_demo_synthetic.csv`
- Safe numeric conversion for N, P, K, pH and Organic Carbon
- Median imputation for numeric values and mode imputation for categorical values
- Duplicate detection and removal with before/after reporting
- Broad range validation with warnings for suspicious soil values
- Explainable IQR outlier counts without automatic outlier deletion
- Derived `NPK_Total` feature and future-model feature/target inventory
- Interactive Data Intelligence page with statistics, distributions, correlation, fertility and regional charts
- Processed output at `data/processed/soil_processed.csv`
- Metadata output at `data/processed/preprocessing_summary.json`

The demo CSV is synthetic development data. It does not represent real agricultural measurements or farm soil, and it must be replaced with a validated real-world dataset before production use.

### Part 3 completed

- Reproducible stratified 80/20 train/test split with `random_state=42`
- Random Forest Classifier with 200 estimators and balanced class weights
- Weighted accuracy, precision, recall, F1 score, classification report and confusion matrix
- Direct feature importance from the trained forest
- Saved model artifact at `models/soil_fertility_model.joblib`
- Saved model card and evaluation metadata at `models/model_metadata.json`
- Soil Health page with model status, evaluation charts and real inference
- Model Insights page with training contract, limitations and feature importance
- Probability output labelled as model confidence, not certainty

The prototype uses only `N`, `P`, `K`, `pH` and `Organic_Carbon` as model inputs. `NPK_Total` is retained in the processed dataset for later analysis but is excluded from the classifier to avoid redundant inputs. The model is trained using demo/synthetic data for prototype validation. Its evaluation does not establish real-world agricultural or laboratory-level accuracy, and predictions should not replace laboratory soil testing.

### Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

### Project structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── model_metadata.json
│   └── soil_fertility_model.joblib
├── modules/
│   ├── analytics.py
│   ├── data_loader.py
│   ├── data_utils.py
│   ├── model.py
│   └── preprocessing.py
└── assets/
```

The current dataset columns are `N`, `P`, `K`, `pH`, `Organic_Carbon`, `Region`, `Crop`, `Soil_Type`, `Moisture`, `Temperature`, `Rainfall`, and `Fertility`. The target is `Fertility`; `N`, `P`, `K`, `pH`, `Organic_Carbon`, environmental values, and engineered `NPK_Total` form the future feature inventory.

Future parts can add validated datasets, trained models, reusable analysis modules and visual assets without changing the dashboard entry point.
