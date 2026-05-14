# WoundAI — Clinical Decision Support for Wound Healing Prediction

> A machine learning system that predicts wound healing time for bedside nurses, with SHAP-based explainability and nursing intervention recommendations.

Built by **Tran Thanh An** — Anesthesia Nurse & ML Lead, Nursing Department, University Medical Center (UMC) Ho Chi Minh City, Vietnam.

---

## Overview

WoundAI is a clinical AI backend that helps nurses estimate how long a wound will take to heal, based on patient data and wound characteristics collected at the bedside. The system goes beyond a simple prediction: it explains *why* a wound may heal slowly and suggests concrete nursing interventions to improve outcomes.

This project was developed as a practical application of machine learning in a real clinical nursing context — bridging the gap between data science and bedside care.

---

## Clinical Context

Wound healing prediction is a high-value nursing problem. Prolonged healing increases infection risk, hospital costs, and patient distress. In Vietnam's tertiary hospitals, wound assessment is largely experience-based. This system provides data-driven support to:

- Estimate healing timeline at first assessment
- Identify modifiable risk factors (e.g., dressing frequency, specialist referral)
- Prioritize patients who need closer monitoring

Wound types covered:
- Surgical wounds (`vet_mo`)
- Pressure ulcers (`loet_ap_luc`)
- Venous ulcers (`loet_tinh_mach`)
- Partial-thickness burns, degree 2 (`bong_do_2`)

---

## Technical Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML Model | Random Forest Regressor (scikit-learn) |
| Explainability | SHAP (TreeExplainer) |
| Database | Supabase (PostgreSQL) |
| Deployment | Render.com |
| Language | Python 3.11+ |

---

## Model

**Algorithm:** Random Forest Regressor
- 200 estimators, max depth 10, min samples leaf 3
- Trained with 5-fold cross-validation
- Evaluated on MAE (Mean Absolute Error in days) and R² score

**Input features (9 clinical variables):**

| Feature | Description |
|---|---|
| `age_group` | Patient age group (18–40 / 41–60 / 61–75 / >75) |
| `diabetes` | Diabetes status (binary) |
| `wound_type` | Type of wound (4 categories) |
| `length_cm` | Wound length in cm |
| `width_cm` | Wound width in cm |
| `depth_cm` | Wound depth in cm |
| `area_cm2` | Wound area (length × width) |
| `dressing_per_week` | Number of dressing changes per week |
| `nurse_specialist` | Whether care is provided by a wound specialist nurse |

**Output:** Predicted healing time in days

**Explainability:** SHAP values identify which factors are driving the prediction for each individual patient, with:
- Clinical explanation of each factor's impact
- Specific nursing intervention recommendations for modifiable factors

---

## API Endpoints

```
POST /predict          — Predict healing days + SHAP explanation
POST /wounds           — Record a new wound case
GET  /stats            — Dashboard statistics (overview, model performance, nurse comparison)
POST /confirm-healed   — Mark a wound as healed (updates actual_days for retraining)
```

Full interactive API docs available at `/docs` when running locally.

---

## Project Structure

```
woundai-backend/
├── main.py              # FastAPI app — all endpoints
├── train_model.py       # Model training + SHAP explainer generation
├── generate_data.py     # Synthetic clinical data generator (300 cases)
├── auto_retrain.py      # Scheduled retraining when new confirmed cases accumulate
├── create_tables.sql    # Supabase database schema
├── model_registry.json  # Model versioning metadata
├── render.yaml          # Render.com deployment config
└── requirements.txt     # Python dependencies
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/tranthanhan1626-boop/woundai-backend.git
cd woundai-backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Fill in SUPABASE_URL and SUPABASE_KEY

# 4. Generate synthetic data (first time)
python generate_data.py

# 5. Train the model
python train_model.py

# 6. Start the API
uvicorn main:app --reload

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## Deployment

Deployed on **Render.com** via `render.yaml`.

Build command: `pip install -r requirements.txt && python3 train_model.py`  
Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Environment variables required: `SUPABASE_URL`, `SUPABASE_KEY`

---

## About the Author

**Tran Thanh An**  
Registered Nurse — Anesthesia & Perioperative Care  
ML Lead, Nursing Department AI Integration Roadmap  
University Medical Center (UMC), Ho Chi Minh City, Vietnam

Active participant in liver and kidney transplant programs. Currently building expertise in machine learning and deep learning for clinical applications, with a focus on AI-assisted decision support in anesthesia and wound care.

- Email: an.tt1@umc.edu.vn  
- Certifications: Supervised ML (DeepLearning.AI × Stanford) · Python for Data Science (IBM) · Data Science Methodology (IBM)
- Currently completing: Advanced Deep Learning Specialization — DeepLearning.AI

---

## License

MIT License — open for academic and research use.
