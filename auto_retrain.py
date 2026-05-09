"""
auto_retrain.py
Tự động train lại mô hình khi đủ 200 ca mới có outcome thật.
Sliding window: luôn dùng 1000 ca mới nhất để train.
Chạy: python3 auto_retrain.py
Render Cron: cấu hình chạy mỗi ngày lúc 2:00 AM
"""

import os, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ── Cấu hình ────────────────────────────────────────────────────────────────
MAX_WINDOW    = 1000   # Tối đa bao nhiêu ca dùng để train
TRIGGER_NEW   = 200    # Đủ bao nhiêu ca mới có outcome thì train lại
REGISTRY_FILE = "model_registry.json"

FEATURES = [
    'age_group', 'diabetes', 'wound_type',
    'length_cm', 'width_cm', 'depth_cm', 'area_cm2',
    'dressing_per_week', 'nurse_specialist',
]

FEATURE_LABELS = {
    'age_group':         'Nhóm tuổi',
    'diabetes':          'Đái tháo đường',
    'wound_type':        'Loại vết thương',
    'length_cm':         'Chiều dài vết thương',
    'width_cm':          'Chiều rộng vết thương',
    'depth_cm':          'Độ sâu vết thương',
    'area_cm2':          'Diện tích vết thương',
    'dressing_per_week': 'Tần suất thay băng',
    'nurse_specialist':  'Điều dưỡng chuyên khoa',
}

CLINICAL_EXPLANATIONS = {
    'diabetes':          {'bad': 'Đường huyết cao làm giảm lưu thông máu, tế bào lành chậm hơn đáng kể.', 'good': 'Không có đái tháo đường — lưu thông máu tốt, tế bào hồi phục nhanh.'},
    'wound_type':        {'bad': 'Loại vết thương này thường lành chậm hơn — cần theo dõi sát hơn.', 'good': 'Loại vết thương thuận lợi để lành, tiên lượng tốt.'},
    'area_cm2':          {'bad': 'Diện tích lớn đòi hỏi nhiều mô hạt hơn để lấp đầy, cần thêm thời gian.', 'good': 'Diện tích nhỏ — mép vết thương gần nhau, dễ liền hơn.'},
    'depth_cm':          {'bad': 'Vết thương sâu cần lành từ đáy lên, mất nhiều thời gian hơn.', 'good': 'Vết thương nông — lành nhanh từ mép vào.'},
    'dressing_per_week': {'bad': 'Tần suất thay băng chưa đủ — tăng lên có thể rút ngắn thời gian lành.', 'good': 'Tần suất thay băng tốt — giữ vết thương sạch và ẩm đúng mức.'},
    'nurse_specialist':  {'bad': 'Điều dưỡng đa khoa chăm sóc — cân nhắc hội chẩn điều dưỡng chuyên khoa.', 'good': 'Điều dưỡng chuyên khoa vết thương — kỹ thuật và sản phẩm được tối ưu.'},
    'age_group':         {'bad': 'Người cao tuổi tốc độ tái tạo tế bào chậm hơn, cần theo dõi kỹ hơn.', 'good': 'Nhóm tuổi trẻ — tốc độ lành tốt, hệ miễn dịch mạnh.'},
    'length_cm':         {'bad': 'Vết thương dài — cần thời gian liền mép theo chiều dài.', 'good': 'Chiều dài trong mức bình thường.'},
    'width_cm':          {'bad': 'Vết thương rộng — diện tích cần lành nhiều hơn.', 'good': 'Chiều rộng trong mức bình thường.'},
}

NURSING_INTERVENTIONS = {
    'diabetes':          'Báo cáo đường huyết cho bác sĩ — kiểm soát HbA1c là yếu tố quan trọng nhất có thể thay đổi được.',
    'dressing_per_week': 'Tăng tần suất thay băng — có thể rút ngắn thêm 3–5 ngày lành.',
    'nurse_specialist':  'Hội chẩn điều dưỡng chuyên khoa vết thương để tối ưu phác đồ.',
    'area_cm2':          'Đánh giá lại sản phẩm băng phù hợp kích thước vết thương.',
    'depth_cm':          'Sử dụng vật liệu lấp đầy (packing) để kích thích mô hạt từ đáy lên.',
    'wound_type':        'Xem lại phác đồ chăm sóc theo loại vết thương — mỗi loại có tiêu chuẩn riêng.',
    'age_group':         'Tăng cường dinh dưỡng: protein và vitamin C hỗ trợ tổng hợp collagen.',
    'length_cm':         'Đảm bảo băng phủ đủ toàn bộ chiều dài vết thương.',
    'width_cm':          'Kiểm tra sản phẩm băng đủ rộng che phủ vết thương.',
}


def load_registry():
    """Đọc thông tin mô hình hiện tại"""
    try:
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    except:
        return {"version": "v0.0", "mae_days": 999, "trained_cases": 0,
                "last_retrain_at": "2020-01-01T00:00:00", "is_current": True}


def save_registry(version, mae, r2, n_cases):
    registry = {
        "version":          version,
        "trained_at":       datetime.now().isoformat(),
        "last_retrain_at":  datetime.now().isoformat(),
        "mae_days":         round(float(mae), 2),
        "r2_score":         round(float(r2), 3),
        "trained_cases":    n_cases,
        "is_current":       True,
        "has_shap":         True,
    }
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    return registry


def count_new_cases(last_retrain_at):
    """Đếm ca MỚI có outcome thật kể từ lần train cuối"""
    result = supabase.table("wounds")\
        .select("id", count="exact")\
        .not_.is_("actual_days", "null")\
        .gt("updated_at", last_retrain_at)\
        .execute()
    return result.count or 0


def load_sliding_window():
    """
    Lấy đúng MAX_WINDOW ca mới nhất có outcome thật.
    Ca cũ hơn bị bỏ qua khi train — nhưng vẫn còn trong DB để kiểm toán.
    """
    print(f"   Lấy {MAX_WINDOW} ca mới nhất từ database...")
    wounds = supabase.table("wounds")\
        .select("*, patients(*)")\
        .not_.is_("actual_days", "null")\
        .order("created_at", desc=True)\
        .limit(MAX_WINDOW)\
        .execute().data

    rows = []
    for w in wounds:
        p = w.get('patients')
        if not p:
            continue
        visits = supabase.table("visits")\
            .select("*").eq("wound_id", w['id'])\
            .order("visit_date").limit(1).execute().data
        if not visits:
            continue
        v = visits[0]
        rows.append({
            'age_group':         p.get('age_group', '41-60'),
            'diabetes':          1 if p.get('diabetes') else 0,
            'wound_type':        w['wound_type'],
            'length_cm':         v['length_cm'],
            'width_cm':          v['width_cm'],
            'depth_cm':          v['depth_cm'],
            'area_cm2':          round(v['length_cm'] * v['width_cm'], 2),
            'dressing_per_week': v['dressing_per_week'],
            'nurse_specialist':  1 if v['nurse_type'] == 'specialist' else 0,
            'actual_days':       w['actual_days'],
        })

    df = pd.DataFrame(rows)
    print(f"   ✅ Sliding window: {len(df)} ca")
    return df


def preprocess(df):
    age_map   = {'18-40': 0, '41-60': 1, '61-75': 2, '>75': 3}
    wound_map = {'vet_mo': 0, 'loet_ap_luc': 1, 'loet_tinh_mach': 2, 'bong_do_2': 3}
    df['age_group']  = df['age_group'].map(age_map).fillna(1)
    df['wound_type'] = df['wound_type'].map(wound_map).fillna(0)
    return df


def train_new_model(df):
    """Train Random Forest mới trên sliding window"""
    X = df[FEATURES]
    y = df['actual_days']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2  = r2_score(y_test, model.predict(X_test))
    cv_mae = -cross_val_score(
        model, X, y, scoring='neg_mean_absolute_error', cv=5
    ).mean()

    # Fit lại toàn bộ data
    model.fit(X, y)
    return model, cv_mae, r2


def get_next_version(current_version):
    """Tăng version: v1.0 → v1.1 → v1.2 ..."""
    try:
        major, minor = current_version.lstrip('v').split('.')
        return f"v{major}.{int(minor)+1}"
    except:
        return "v1.1"


def deploy_new_model(model, mae, r2, n_cases, new_version):
    """Lưu model mới và tất cả metadata"""
    explainer = shap.TreeExplainer(model)

    joblib.dump(model,                'model_v1.pkl')
    joblib.dump(explainer,            'shap_explainer.pkl')
    joblib.dump(FEATURES,             'features.pkl')
    joblib.dump(FEATURE_LABELS,       'feature_labels.pkl')
    joblib.dump(CLINICAL_EXPLANATIONS,'clinical_explanations.pkl')
    joblib.dump(NURSING_INTERVENTIONS,'nursing_interventions.pkl')

    registry = save_registry(new_version, mae, r2, n_cases)
    print(f"   ✅ Đã deploy {new_version} · Sai số: {mae:.1f} ngày · R²: {r2:.3f}")
    return registry


def run():
    print("\n" + "="*50)
    print(f"🔄 Auto-retrain bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*50)

    # 1. Đọc registry hiện tại
    registry = load_registry()
    current_mae     = registry.get('mae_days', 999)
    current_version = registry.get('version', 'v1.0')
    last_retrain    = registry.get('last_retrain_at', '2020-01-01T00:00:00')

    print(f"\n📋 Mô hình hiện tại: {current_version} · Sai số: {current_mae} ngày")

    # 2. Đếm ca mới có outcome
    new_cases = count_new_cases(last_retrain)
    print(f"📊 Ca mới có outcome: {new_cases} / {TRIGGER_NEW} cần thiết")

    if new_cases < TRIGGER_NEW:
        print(f"\n⏳ Chưa đủ ca mới — cần thêm {TRIGGER_NEW - new_cases} ca nữa")
        print("   Hệ thống giữ nguyên mô hình hiện tại")
        return False

    print(f"\n✅ Đủ {new_cases} ca mới — bắt đầu retrain!")

    # 3. Lấy sliding window
    df = load_sliding_window()
    if len(df) < 50:
        print("❌ Không đủ dữ liệu để train (cần tối thiểu 50 ca)")
        return False

    df = preprocess(df)

    # 4. Train mô hình mới
    print(f"\n🔄 Train mô hình mới trên {len(df)} ca...")
    model_new, mae_new, r2_new = train_new_model(df)
    print(f"   Sai số mô hình mới: {mae_new:.1f} ngày · R²: {r2_new:.3f}")
    print(f"   Sai số mô hình cũ:  {current_mae:.1f} ngày")

    # 5. So sánh và quyết định deploy
    if mae_new < current_mae:
        new_version = get_next_version(current_version)
        print(f"\n🎉 Mô hình mới tốt hơn! Deploy {new_version}...")
        deploy_new_model(model_new, mae_new, r2_new, len(df), new_version)
        print(f"\n✅ Hoàn thành! {current_version} → {new_version}")
        print(f"   Cải thiện sai số: {current_mae - mae_new:.1f} ngày")
        return True
    else:
        improvement_needed = mae_new - current_mae
        print(f"\n⚠️  Mô hình mới chưa tốt hơn ({improvement_needed:+.1f} ngày)")
        print(f"   Giữ nguyên {current_version}")

        # Kiểm tra xem thực tế có đang xấu đi không (cảnh báo lâm sàng)
        if mae_new > current_mae * 1.2:
            print(f"\n🚨 CẢNH BÁO LÂM SÀNG: Sai số tăng >20%")
            print(f"   Có thể quy trình chăm sóc đang có vấn đề")
            print(f"   Đề nghị kiểm tra lại phác đồ chăm sóc!")
        return False


if __name__ == "__main__":
    run()
