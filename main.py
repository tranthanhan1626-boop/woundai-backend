"""
main.py
FastAPI server — WoundAI với SHAP giải thích cho điều dưỡng
Chạy: python3 main.py
"""

import os, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI(
    title="WoundAI API",
    description="Hệ thống dự báo thời gian lành vết thương cho điều dưỡng",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Load tất cả model và metadata khi server khởi động ──────────────────────
try:
    model       = joblib.load("model_v1.pkl")
    explainer   = joblib.load("shap_explainer.pkl")
    FEATURES    = joblib.load("features.pkl")
    F_LABELS    = joblib.load("feature_labels.pkl")
    CLINICAL    = joblib.load("clinical_explanations.pkl")
    NURSING     = joblib.load("nursing_interventions.pkl")
    print("✅ Đã load model_v1.pkl + SHAP explainer")
except FileNotFoundError as e:
    print(f"❌ Thiếu file: {e} — Hãy chạy train_model.py trước")
    model = explainer = None


# ── Schema đầu vào ───────────────────────────────────────────────────────────
class WoundInput(BaseModel):
    age_group:         str    # '18-40' | '41-60' | '61-75' | '>75'
    diabetes:          bool
    wound_type:        str    # 'vet_mo' | 'loet_ap_luc' | 'loet_tinh_mach' | 'bong_do_2'
    length_cm:         float
    width_cm:          float
    depth_cm:          float
    dressing_per_week: int
    nurse_type:        str    # 'specialist' | 'general'
    patient_name:      str = "Ẩn danh"
    wound_id:          str = None


# ── Tiền xử lý ───────────────────────────────────────────────────────────────
def preprocess(data: WoundInput) -> pd.DataFrame:
    age_map   = {'18-40': 0, '41-60': 1, '61-75': 2, '>75': 3}
    wound_map = {'vet_mo': 0, 'loet_ap_luc': 1, 'loet_tinh_mach': 2, 'bong_do_2': 3}
    row = {
        'age_group':         age_map.get(data.age_group, 1),
        'diabetes':          1 if data.diabetes else 0,
        'wound_type':        wound_map.get(data.wound_type, 0),
        'length_cm':         data.length_cm,
        'width_cm':          data.width_cm,
        'depth_cm':          data.depth_cm,
        'area_cm2':          round(data.length_cm * data.width_cm, 2),
        'dressing_per_week': data.dressing_per_week,
        'nurse_specialist':  1 if data.nurse_type == 'specialist' else 0,
    }
    return pd.DataFrame([row], columns=FEATURES)


def get_risk(days: int) -> dict:
    if days <= 21:
        return {"level": "low",    "label": "Nguy cơ thấp",       "note": "Lành bình thường — duy trì kế hoạch hiện tại"}
    elif days <= 42:
        return {"level": "medium", "label": "Nguy cơ trung bình", "note": "Theo dõi sát — đánh giá lại sau 1 tuần"}
    else:
        return {"level": "high",   "label": "Nguy cơ cao",        "note": "Cần hội chẩn điều dưỡng chuyên khoa hoặc bác sĩ"}


def build_shap_result(X_df: pd.DataFrame) -> dict:
    """
    Tính SHAP values và dịch sang ngôn ngữ lâm sàng cho điều dưỡng.
    Trả về danh sách yếu tố làm chậm, yếu tố giúp lành, và gợi ý can thiệp.
    """
    shap_vals = explainer.shap_values(X_df)[0]  # array 9 phần tử
    row       = X_df.iloc[0]

    factors = []
    for feat, shap_val in zip(FEATURES, shap_vals):
        days_impact = round(float(shap_val), 1)
        direction   = 'bad' if shap_val > 0 else 'good'

        # Lấy giải thích lâm sàng
        clinical_exp = CLINICAL.get(feat, {}).get(direction, '')

        factors.append({
            'feature':       feat,
            'label':         F_LABELS.get(feat, feat),
            'days_impact':   days_impact,         # + = làm chậm, - = giúp lành
            'direction':     direction,
            'explanation':   clinical_exp,
            'intervention':  NURSING.get(feat, '') if direction == 'bad' else '',
        })

    # Sắp xếp: yếu tố ảnh hưởng nhiều nhất lên đầu
    factors.sort(key=lambda x: abs(x['days_impact']), reverse=True)

    # Tách thành 2 nhóm
    slowing_factors = [f for f in factors if f['direction'] == 'bad'  and abs(f['days_impact']) > 0.5]
    helping_factors = [f for f in factors if f['direction'] == 'good' and abs(f['days_impact']) > 0.5]

    # Gợi ý can thiệp: chỉ lấy top 3 yếu tố có thể thay đổi được
    changeable = ['dressing_per_week', 'nurse_specialist', 'diabetes']
    interventions = []
    for f in slowing_factors:
        if f['feature'] in changeable and f['intervention']:
            interventions.append({
                'factor':       f['label'],
                'action':       f['intervention'],
                'days_saving':  abs(f['days_impact']),
            })

    return {
        'slowing_factors': slowing_factors[:4],   # top 4 yếu tố làm chậm
        'helping_factors': helping_factors[:3],   # top 3 yếu tố giúp lành
        'interventions':   interventions[:3],     # top 3 gợi ý can thiệp
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"system": "WoundAI", "status": "running",
            "model": "v1.0", "time": datetime.now().isoformat()}

@app.get("/health")
def health():
    return {
        "server":   "ok",
        "model":    "loaded" if model    else "not loaded",
        "shap":     "loaded" if explainer else "not loaded",
    }


@app.post("/predict")
def predict(data: WoundInput):
    """
    Dự báo thời gian lành + giải thích SHAP cho điều dưỡng.
    Trả về: số ngày, nguy cơ, yếu tố ảnh hưởng, gợi ý can thiệp.
    """
    if model is None or explainer is None:
        raise HTTPException(503, "Mô hình chưa load. Chạy train_model.py trước.")

    X_df           = preprocess(data)
    predicted_days = int(round(model.predict(X_df)[0]))
    confidence_low = max(5, int(predicted_days * 0.80))
    confidence_high = int(predicted_days * 1.20)
    risk           = get_risk(predicted_days)
    shap_result    = build_shap_result(X_df)

    # Lưu vào bảng predictions nếu có wound_id hợp lệ
    if data.wound_id and len(data.wound_id) > 10:
        try:
            supabase.table("predictions").insert({
                "wound_id":       data.wound_id,
                "predicted_days": predicted_days,
                "confidence_low": confidence_low,
                "confidence_high":confidence_high,
                "model_version":  "v1.0",
            }).execute()
        except Exception as e:
            print(f"⚠️ Không lưu prediction: {e}")

    return {
        # Kết quả chính
        "predicted_days":   predicted_days,
        "confidence_low":   confidence_low,
        "confidence_high":  confidence_high,
        "risk":             risk,
        "model_version":    "v1.0",

        # SHAP — giải thích cho điều dưỡng
        "shap": shap_result,

        # Tóm tắt đầu vào
        "input_summary": {
            "wound_type":   data.wound_type,
            "area_cm2":     round(data.length_cm * data.width_cm, 1),
            "diabetes":     data.diabetes,
            "nurse_type":   data.nurse_type,
        }
    }


@app.get("/stats")
def stats():
    try:
        n_patients = len(supabase.table("patients").select("id").execute().data)
        n_wounds   = len(supabase.table("wounds").select("id").execute().data)
        n_healed   = len(supabase.table("wounds").select("id")
                         .not_.is_("actual_days", "null").execute().data)
        with open("model_registry.json") as f:
            registry = json.load(f)
        return {
            "database": {"patients": n_patients, "wounds": n_wounds, "healed": n_healed},
            "model": registry,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Khởi động ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 WoundAI API đang khởi động...")
    print("   Địa chỉ local: http://localhost:8000")
    print("   Tài liệu API:  http://localhost:8000/docs")
    print("   Nhấn Ctrl+C để dừng\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
