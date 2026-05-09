"""
generate_data.py
Tạo 300 ca vết thương giả lập có tính thực tế
và lưu vào Supabase database
"""

import os
import random
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

random.seed(42)
np.random.seed(42)

# ============================================
# Cấu hình dữ liệu thực tế lâm sàng
# ============================================

WOUND_TYPES = ['vet_mo', 'loet_ap_luc', 'loet_tinh_mach', 'bong_do_2']
AGE_GROUPS  = ['18-40', '41-60', '61-75', '>75']

# Số ngày lành trung bình theo loại vết thương (thực tế lâm sàng)
HEAL_BASE = {
    'vet_mo':          14,
    'loet_ap_luc':     45,
    'loet_tinh_mach':  35,
    'bong_do_2':       21,
}

def calc_heal_days(wound_type, age_group, diabetes,
                   length, width, depth,
                   dressing_per_week, nurse_type):
    """
    Tính số ngày lành dựa trên các yếu tố lâm sàng
    Công thức mô phỏng thực tế — không phải số ngẫu nhiên thuần túy
    """
    days = HEAL_BASE[wound_type]

    # Tuổi cao → lành chậm hơn
    age_factor = {'18-40': 0, '41-60': 5, '61-75': 12, '>75': 20}
    days += age_factor[age_group]

    # Đái tháo đường → lành chậm hơn đáng kể
    if diabetes:
        days += int(days * 0.45)

    # Kích thước vết thương càng lớn → càng lâu
    area = length * width
    days += int(area * 0.8)
    days += int(depth * 3)

    # Thay băng nhiều hơn → lành nhanh hơn
    days -= (dressing_per_week - 1) * 2

    # Điều dưỡng chuyên khoa → lành nhanh hơn
    if nurse_type == 'specialist':
        days -= 4

    # Thêm nhiễu ngẫu nhiên thực tế (±20%)
    noise = np.random.normal(0, days * 0.2)
    days = max(5, int(days + noise))

    return days


def generate_patients(n=300):
    patients = []
    for i in range(n):
        age_group = random.choices(
            AGE_GROUPS,
            weights=[15, 35, 35, 15]   # phân bố thực tế bệnh viện
        )[0]

        # Người cao tuổi và béo phì dễ đái tháo đường hơn
        dm_prob = 0.15
        if age_group in ['61-75', '>75']:
            dm_prob = 0.40

        patients.append({
            "full_name":   f"Bệnh nhân {i+1:03d}",
            "age_group":   age_group,
            "gender":      random.choice(['male', 'female']),
            "diabetes":    random.random() < dm_prob,
            "hospital_id": f"BV-{random.randint(1,3):02d}",
        })
    return patients


def generate_wound(patient_id, start_date):
    wound_type       = random.choices(
        WOUND_TYPES,
        weights=[30, 25, 25, 20]
    )[0]
    length           = round(random.uniform(1.0, 8.0), 1)
    width            = round(random.uniform(0.5, length), 1)
    depth            = round(random.uniform(0.1, 2.0), 1)
    dressing_per_week= random.randint(1, 7)
    nurse_type       = random.choices(
        ['specialist', 'general'],
        weights=[60, 40]
    )[0]

    return {
        "patient_id":    patient_id,
        "wound_type":    wound_type,
        "location":      random.choice([
            'sacrum', 'heel', 'ankle', 'leg', 'abdomen', 'back'
        ]),
        "created_date":  str(start_date),
        "length":        length,
        "width":         width,
        "depth":         depth,
        "dressing_per_week": dressing_per_week,
        "nurse_type":    nurse_type,
    }


def main():
    print("🔄 Bắt đầu tạo dữ liệu giả lập...")

    # 1. Tạo bệnh nhân
    print("   Tạo 300 bệnh nhân...")
    patients_data = generate_patients(300)

    # Chèn từng lô 50 để tránh timeout
    patient_ids = []
    for i in range(0, len(patients_data), 50):
        batch = patients_data[i:i+50]
        res = supabase.table("patients").insert(batch).execute()
        for p in res.data:
            patient_ids.append(p['id'])
    print(f"   ✅ Đã tạo {len(patient_ids)} bệnh nhân")

    # 2. Tạo vết thương và lần thăm khám
    print("   Tạo vết thương và lịch thăm khám...")
    wounds_inserted = 0

    # Trải ngày nhập viện từ 2020 đến 2025
    start = date(2020, 1, 1)
    end   = date(2024, 12, 31)
    span  = (end - start).days

    for i, patient_id in enumerate(patient_ids):
        # Lấy thông tin bệnh nhân để tính ngày lành
        p = patients_data[i]

        # Ngày nhập viện ngẫu nhiên trong khoảng 2020-2024
        admit_date = start + timedelta(days=random.randint(0, span))

        wound_info = generate_wound(patient_id, admit_date)

        # Tính số ngày lành thực tế
        heal_days = calc_heal_days(
            wound_type        = wound_info['wound_type'],
            age_group         = p['age_group'],
            diabetes          = p['diabetes'],
            length            = wound_info['length'],
            width             = wound_info['width'],
            depth             = wound_info['depth'],
            dressing_per_week = wound_info['dressing_per_week'],
            nurse_type        = wound_info['nurse_type'],
        )

        healed_date = admit_date + timedelta(days=heal_days)

        # Chèn vào bảng wounds
        wound_res = supabase.table("wounds").insert({
            "patient_id":        wound_info['patient_id'],
            "wound_type":        wound_info['wound_type'],
            "location":          wound_info['location'],
            "created_date":      wound_info['created_date'],
            "actual_healed_date": str(healed_date),
        }).execute()

        wound_id = wound_res.data[0]['id']

        # Tạo 3-5 lần thăm khám (longitudinal)
        n_visits = random.randint(3, 5)
        for v in range(n_visits):
            visit_date = admit_date + timedelta(
                days=int(heal_days * v / n_visits)
            )
            # Kích thước giảm dần theo thời gian (vết thương đang lành)
            shrink = 1 - (v / n_visits) * 0.7
            supabase.table("visits").insert({
                "wound_id":           wound_id,
                "visit_date":         str(visit_date),
                "length_cm":          round(wound_info['length'] * shrink, 1),
                "width_cm":           round(wound_info['width']  * shrink, 1),
                "depth_cm":           round(wound_info['depth']  * shrink, 1),
                "dressing_per_week":  wound_info['dressing_per_week'],
                "nurse_type":         wound_info['nurse_type'],
            }).execute()

        wounds_inserted += 1
        if wounds_inserted % 50 == 0:
            print(f"   ... {wounds_inserted}/300 ca")

    print(f"\n✅ Hoàn thành! Đã tạo:")
    print(f"   - {len(patient_ids)} bệnh nhân")
    print(f"   - {wounds_inserted} vết thương")
    print(f"   - ~{wounds_inserted * 4} lần thăm khám")
    print(f"\n👉 Bước tiếp theo: chạy python3 train_model.py")


if __name__ == "__main__":
    main()
