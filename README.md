# AGROSOIL AI

## Intelligent Soil Health Analysis & Fertilizer Recommendation System

AGROSOIL AI is a premium Streamlit product experience for **DataXcelerate 2026, PS22**. Part 2 adds a transparent soil-data loading, validation, preprocessing and analytics foundation for future ML work.

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
├── modules/
│   ├── analytics.py
│   ├── data_loader.py
│   ├── data_utils.py
│   └── preprocessing.py
└── assets/
```

The current dataset columns are `N`, `P`, `K`, `pH`, `Organic_Carbon`, `Region`, `Crop`, `Soil_Type`, `Moisture`, `Temperature`, `Rainfall`, and `Fertility`. The target is `Fertility`; `N`, `P`, `K`, `pH`, `Organic_Carbon`, environmental values, and engineered `NPK_Total` form the future feature inventory.

Future parts can add validated datasets, trained models, reusable analysis modules and visual assets without changing the dashboard entry point.
