# AGROSOIL AI

## Intelligent Soil Health Analysis & Fertilizer Recommendation System

AGROSOIL AI is a premium Streamlit product experience for **DataXcelerate 2026, PS22**. Part 1 establishes the dashboard, navigation and product foundation for future soil intelligence modules.

### Part 1 scope

- Premium AgriTech dashboard with responsive glassmorphism styling
- Hero landing screen with placeholder KPI values
- Soil report, soil image and farmer experience entry points
- Soil intelligence feature overview
- Visual AI workflow from input to recommendation
- About page and polished placeholders for future modules
- No fake datasets, predictions, recommendations or API keys

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
├── models/
├── modules/
└── assets/
```

Future parts can add datasets, trained models, reusable analysis modules and visual assets without changing the dashboard entry point.
