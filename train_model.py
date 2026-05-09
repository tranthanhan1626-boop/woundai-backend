"""
train_model.py
Train Random Forest + tạo SHAP explainer
Lưu: model_v1.pkl · shap_explainer.pkl · features.pkl
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

FEATURES = [
    'age_group', 'diabetes', 'wound_type',
    'length_cm', 'width_cm', 'depth_cm', 'area_cm2',
    'dressing_per_week', 'nurse_specialist',
]

# Nhãn hiển thị tiếng Việt cho điều dưỡng
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

# Giải thích lâm sàng: yếu tố làm chậm (bad) hoặc giúp lành (good)
CLINICAL_EXPLANATIONS = {
    'diabetes': {
        'bad':  'Đường huyết cao làm giảm lưu thông máu, tế bào lành chậm hơn đáng kể.',
        'good': 'Không có đái tháo đường — lưu thông máu tốt, tế bào hồi phục nhanh.',
    },
    'wound_type': {
        'bad':  'Loại vết thương này thường lành chậm hơn — cần theo dõi sát hơn.',
        'good': 'Loại vết thương thuận lợi để lành, tiên lượng tốt.',
    },
    'area_cm2': {
        'bad':  'Diện tích lớn đòi hỏi nhiều mô hạt hơn để lấp đầy, cần thêm thời gian.',
        'good': 'Diện tích nhỏ — mép vết thương gần nhau, dễ liền hơn.',
    },
    'depth_cm': {
        'bad':  'Vết thương sâu cần lành từ đáy lên, mất nhiều thời gian hơn.',
        'good': 'Vết thương nông — lành nhanh từ mép vào.',
    },
    'dressing_per_week': {
        'bad':  'Tần suất thay băng chưa đủ — tăng lên có thể rút ngắn thời gian lành.',
        'good': 'Tần suất thay băng tốt — giữ vết thương sạch và ẩm đúng mức.',
    },
    'nurse_specialist': {
        'bad':  'Điều dưỡng đa khoa chăm sóc — cân nhắc hội chẩn điều dưỡng chuyên khoa.',
        'good': 'Điều dưỡng chuyên khoa vết thương — kỹ thuật và sản phẩm được tối ưu.',
    },
    'age_group': {
        'bad':  'Người cao tuổi tốc độ tái tạo tế bào chậm hơn, cần theo dõi kỹ hơn.',
        'good': 'Nhóm tuổi trẻ — tốc độ lành tốt, hệ miễn dịch mạnh.',
    },
    'length_cm': {
        'bad':  'Vết thương dài — cần thời gian liền mép theo chiều dài.',
        'good': 'Chiều dài trong mức bình thường.',
    },
    'width_cm': {
        'bad':  'Vết thương rộng — diện tích cần lành nhiều hơn.',
        'good': 'Chiều rộng trong mức bình thường.',
    },
}

# Gợi ý can thiệp điều dưỡng theo từng yếu tố làm chậm
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


def load_data():
    print("🔄 Đang tải dữ liệu từ Supabase...")
    wounds = supabase.table("wounds")\
        .select("*, patients(*)")\
        .not_.is_("actual_days", "null")\
        .execute().data

    if not wounds:
        print("❌ Chưa có dữ liệu. Chạy generate_data.py trước.")
        return None

    rows = []
    for w in wounds:
        p = w['patients']
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
    print(f"   ✅ Tải được {len(df)} ca")
    return df


def preprocess(df):
    age_map   = {'18-40': 0, '41-60': 1, '61-75': 2, '>75': 3}
    wound_map = {'vet_mo': 0, 'loet_ap_luc': 1, 'loet_tinh_mach': 2, 'bong_do_2': 3}
    df['age_group']  = df['age_group'].map(age_map).fillna(1)
    df['wound_type'] = df['wound_type'].map(wound_map).fillna(0)
    return df


def train(df):
    X = df[FEATURES]
    y = df['actual_days']
    print(f"\n🔄 Train mô hình ({len(X)} ca · {len(FEATURES)} đặc trưng)...")

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

    print(f"\n📊 Kết quả mô hình:")
    print(f"   Sai số trung bình (CV 5-fold): {cv_mae:.1f} ngày")
    print(f"   R² score:                      {r2:.3f}")

    # Fit lại toàn bộ data trước khi lưu
    model.fit(X, y)
    return model, cv_mae, r2, X


def build_shap(model, X):
    print("\n🔄 Tạo SHAP explainer (lần đầu mất 1–2 phút)...")
    explainer = shap.TreeExplainer(model)
    # Kiểm tra chạy được không
    _ = explainer.shap_values(X.iloc[:5])
    print("   ✅ SHAP explainer hoạt động tốt")
    return explainer


def save_all(model, explainer, mae, r2):
    joblib.dump(model,     'model_v1.pkl')
    joblib.dump(explainer, 'shap_explainer.pkl')
    joblib.dump(FEATURES,  'features.pkl')

    # Lưu thêm metadata cho API dùng
    joblib.dump(FEATURE_LABELS,       'feature_labels.pkl')
    joblib.dump(CLINICAL_EXPLANATIONS,'clinical_explanations.pkl')
    joblib.dump(NURSING_INTERVENTIONS,'nursing_interventions.pkl')

    registry = {
        "version":    "v1.0",
        "trained_at": datetime.now().isoformat(),
        "mae_days":   round(float(mae), 2),
        "r2_score":   round(float(r2), 3),
        "is_current": True,
        "has_shap":   True,
    }
    with open('model_registry.json', 'w') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Đã lưu:")
    print(f"   model_v1.pkl · shap_explainer.pkl · features.pkl")
    print(f"   feature_labels.pkl · clinical_explanations.pkl · nursing_interventions.pkl")
    print(f"   model_registry.json")
    print(f"\n🎉 Mô hình v1.0 + SHAP sẵn sàng!")
    print(f"   Sai số: {mae:.1f} ngày · R²: {r2:.3f}")
    print(f"\n👉 Bước tiếp theo: python3 main.py")


def main():
    df = load_data()
    if df is None:
        return
    df       = preprocess(df)
    model, mae, r2, X = train(df)
    explainer = build_shap(model, X)
    save_all(model, explainer, mae, r2)


if __name__ == "__main__":
    main()
