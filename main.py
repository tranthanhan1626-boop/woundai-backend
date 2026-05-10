"""
main.py
FastAPI server — WoundAI với SHAP + xác nhận vết thương lành
"""

import os, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

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

# ── Load model khi khởi động ─────────────────────────────────────────────────
try:
    model     = joblib.load("model_v1.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    FEATURES  = joblib.load("features.pkl")
    F_LABELS  = joblib.load("feature_labels.pkl")
    CLINICAL  = joblib.load("clinical_explanations.pkl")
    NURSING   = joblib.load("nursing_interventions.pkl")
    print("✅ Đã load model_v1.pkl + SHAP explainer")
except FileNotFoundError as e:
    print(f"❌ Thiếu file: {e}")
    model = explainer = None


# ── Schemas ──────────────────────────────────────────────────────────────────
class WoundInput(BaseModel):
    age_group:         str
    diabetes:          bool
    wound_type:        str
    length_cm:         float
    width_cm:          float
    depth_cm:          float
    dressing_per_week: int
    nurse_type:        str
    patient_name:      str = "Ẩn danh"
    wound_id:          Optional[str] = None


class HealConfirmInput(BaseModel):
    wound_id:         str           # UUID của vết thương
    actual_healed_date: str         # Ngày lành thật "YYYY-MM-DD"
    nurse_note:       Optional[str] = ""  # Ghi chú của điều dưỡng


class NewCaseInput(BaseModel):
    """Nhập ca mới — tạo patient + wound + visit cùng lúc"""
    # Thông tin bệnh nhân
    patient_name:      str
    age_group:         str
    diabetes:          bool
    gender:            str = "unknown"

    # Thông tin vết thương
    wound_type:        str
    location:          str = ""
    length_cm:         float
    width_cm:          float
    depth_cm:          float
    dressing_per_week: int
    nurse_type:        str


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
    shap_vals = explainer.shap_values(X_df)[0]
    factors   = []
    for feat, shap_val in zip(FEATURES, shap_vals):
        days_impact = round(float(shap_val), 1)
        direction   = 'bad' if shap_val > 0 else 'good'
        factors.append({
            'feature':      feat,
            'label':        F_LABELS.get(feat, feat),
            'days_impact':  days_impact,
            'direction':    direction,
            'explanation':  CLINICAL.get(feat, {}).get(direction, ''),
            'intervention': NURSING.get(feat, '') if direction == 'bad' else '',
        })
    factors.sort(key=lambda x: abs(x['days_impact']), reverse=True)

    slowing  = [f for f in factors if f['direction'] == 'bad'  and abs(f['days_impact']) > 0.5]
    helping  = [f for f in factors if f['direction'] == 'good' and abs(f['days_impact']) > 0.5]
    changeable = ['dressing_per_week', 'nurse_specialist', 'diabetes']
    interventions = [
        {'factor': f['label'], 'action': f['intervention'], 'days_saving': abs(f['days_impact'])}
        for f in slowing if f['feature'] in changeable and f['intervention']
    ]
    return {
        'slowing_factors': slowing[:4],
        'helping_factors': helping[:3],
        'interventions':   interventions[:3],
    }


def get_retrain_status() -> dict:
    """Trạng thái tiến độ đến lần retrain tiếp theo"""
    try:
        with open("model_registry.json") as f:
            reg = json.load(f)
        last_retrain = reg.get("last_retrain_at", "2020-01-01T00:00:00")
        result = supabase.table("wounds")\
            .select("id", count="exact")\
            .not_.is_("actual_days", "null")\
            .gt("updated_at", last_retrain)\
            .execute()
        new_cases = result.count or 0
        return {
            "new_cases":     new_cases,
            "trigger_at":    200,
            "progress_pct":  min(100, round(new_cases / 200 * 100)),
            "model_version": reg.get("version", "v1.0"),
            "model_mae":     reg.get("mae_days", 0),
        }
    except:
        return {"new_cases": 0, "trigger_at": 200, "progress_pct": 0}


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


@app.post("/cases")
def create_case(data: NewCaseInput):
    """
    Tạo ca mới: patient + wound + visit đầu tiên.
    Trả về wound_id để dùng cho /predict và /confirm-healed.
    """
    try:
        # 1. Tạo patient
        p_res = supabase.table("patients").insert({
            "full_name":   data.patient_name,
            "age_group":   data.age_group,
            "gender":      data.gender,
            "diabetes":    data.diabetes,
        }).execute()
        patient_id = p_res.data[0]['id']

        # 2. Tạo wound
        w_res = supabase.table("wounds").insert({
            "patient_id":   patient_id,
            "wound_type":   data.wound_type,
            "location":     data.location,
            "created_date": str(date.today()),
        }).execute()
        wound_id = w_res.data[0]['id']

        # 3. Tạo visit đầu tiên
        supabase.table("visits").insert({
            "wound_id":           wound_id,
            "visit_date":         str(date.today()),
            "length_cm":          data.length_cm,
            "width_cm":           data.width_cm,
            "depth_cm":           data.depth_cm,
            "dressing_per_week":  data.dressing_per_week,
            "nurse_type":         data.nurse_type,
        }).execute()

        return {
            "success":    True,
            "wound_id":   wound_id,
            "patient_id": patient_id,
            "message":    "Ca mới đã được tạo thành công",
        }
    except Exception as e:
        raise HTTPException(500, f"Lỗi tạo ca: {str(e)}")


@app.post("/predict")
def predict(data: WoundInput):
    """Dự báo + SHAP giải thích cho điều dưỡng"""
    if model is None or explainer is None:
        raise HTTPException(503, "Mô hình chưa load.")

    X_df           = preprocess(data)
    predicted_days = int(round(model.predict(X_df)[0]))
    confidence_low = max(5, int(predicted_days * 0.80))
    confidence_high= int(predicted_days * 1.20)
    risk           = get_risk(predicted_days)
    shap_result    = build_shap_result(X_df)

    if data.wound_id and len(data.wound_id) > 10:
        try:
            supabase.table("predictions").insert({
                "wound_id":        data.wound_id,
                "predicted_days":  predicted_days,
                "confidence_low":  confidence_low,
                "confidence_high": confidence_high,
                "model_version":   "v1.0",
            }).execute()
        except Exception as e:
            print(f"⚠️ Không lưu prediction: {e}")

    return {
        "predicted_days":  predicted_days,
        "confidence_low":  confidence_low,
        "confidence_high": confidence_high,
        "risk":            risk,
        "model_version":   "v1.0",
        "shap":            shap_result,
        "input_summary": {
            "wound_type": data.wound_type,
            "area_cm2":   round(data.length_cm * data.width_cm, 1),
            "diabetes":   data.diabetes,
            "nurse_type": data.nurse_type,
        }
    }


@app.post("/confirm-healed")
def confirm_healed(data: HealConfirmInput):
    """
    Điều dưỡng xác nhận vết thương đã lành thật.
    Lưu actual_healed_date → tính actual_days → dùng để retrain.
    """
    try:
        # Lấy ngày tạo vết thương
        wound = supabase.table("wounds")\
            .select("created_date, actual_healed_date")\
            .eq("id", data.wound_id)\
            .execute().data

        if not wound:
            raise HTTPException(404, "Không tìm thấy vết thương")

        w = wound[0]
        if w.get('actual_healed_date'):
            return {"success": False, "message": "Vết thương này đã được xác nhận lành trước đó"}

        # Tính số ngày lành thật
        created  = date.fromisoformat(str(w['created_date']))
        healed   = date.fromisoformat(data.actual_healed_date)
        actual_days = (healed - created).days

        if actual_days < 0:
            raise HTTPException(400, "Ngày lành không thể trước ngày tạo vết thương")

        # Cập nhật vào database
        supabase.table("wounds").update({
            "actual_healed_date": data.actual_healed_date,
            "actual_days":        actual_days,
            "notes":              data.nurse_note or "",
            "updated_at":         datetime.now().isoformat(),
        }).eq("id", data.wound_id).execute()

        # Kiểm tra có đủ 200 ca mới để trigger retrain không
        retrain_status = get_retrain_status()
        trigger_retrain = retrain_status['new_cases'] >= 200

        # Nếu đủ → chạy retrain ngay
        retrain_message = ""
        if trigger_retrain:
            try:
                from auto_retrain import run as do_retrain
                success = do_retrain()
                retrain_message = f"✅ Đã train lại mô hình mới!" if success else "ℹ️ Mô hình cũ vẫn tốt hơn, giữ nguyên."
            except Exception as e:
                retrain_message = f"⚠️ Retrain lỗi: {str(e)}"

        return {
            "success":         True,
            "wound_id":        data.wound_id,
            "actual_days":     actual_days,
            "healed_date":     data.actual_healed_date,
            "message":         f"Đã xác nhận lành sau {actual_days} ngày",
            "retrain_status":  retrain_status,
            "retrain_message": retrain_message,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Lỗi xác nhận: {str(e)}")


@app.get("/retrain-status")
def retrain_status():
    """Trạng thái tiến độ đến lần retrain tiếp theo"""
    return get_retrain_status()


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
            "model":    registry,
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/wounds")
def get_wounds():
    """Lấy danh sách tất cả vết thương + tên bệnh nhân để hiển thị lịch sử"""
    try:
        wounds = supabase.table("wounds")\
            .select("id, patient_id, wound_type, location, created_date, actual_healed_date, actual_days, notes")\
            .order("created_date", desc=True)\
            .execute().data

        patients = supabase.table("patients")\
            .select("id, full_name")\
            .execute().data
        patient_map = {p["id"]: p["full_name"] for p in patients}

        for w in wounds:
            w["patient_name"] = patient_map.get(w["patient_id"], "Không rõ")
            if w["actual_healed_date"]:
                w["status"] = "healed"
            else:
                created = date.fromisoformat(str(w["created_date"]))
                w["days_so_far"] = (date.today() - created).days
                w["status"] = "active"

        return {"success": True, "wounds": wounds}
    except Exception as e:
        raise HTTPException(500, f"Lỗi lấy danh sách vết thương: {str(e)}")
@app.get("/wounds/{wound_id}/visits")
def get_visits(wound_id: str):
    """Lấy danh sách các lần khám của một vết thương"""
    try:
        visits = supabase.table("visits")\
            .select("id, visit_date, length_cm, width_cm, depth_cm, dressing_per_week, nurse_type")\
            .eq("wound_id", wound_id)\
            .order("visit_date", desc=True)\
            .execute().data

        return {"success": True, "visits": visits}
    except Exception as e:
        raise HTTPException(500, f"Lỗi lấy lịch sử khám: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 WoundAI API đang khởi động...")
    print("   Địa chỉ local: http://localhost:8000")
    print("   Tài liệu API:  http://localhost:8000/docs")
    print("   Nhấn Ctrl+C để dừng\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
