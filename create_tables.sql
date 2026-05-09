-- ============================================
-- WOUNDAI · Tạo cấu trúc bảng database
-- Chạy file này trong Supabase SQL Editor
-- ============================================

-- Bảng 1: Bệnh nhân
CREATE TABLE IF NOT EXISTS patients (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    full_name TEXT NOT NULL,
    age_group TEXT CHECK (age_group IN ('18-40', '41-60', '61-75', '>75')),
    gender TEXT CHECK (gender IN ('male', 'female')),
    diabetes BOOLEAN DEFAULT FALSE,
    hospital_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bảng 2: Vết thương (mỗi bệnh nhân có thể có nhiều vết thương)
CREATE TABLE IF NOT EXISTS wounds (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    wound_type TEXT CHECK (wound_type IN (
        'vet_mo',        -- Vết mổ
        'loet_ap_luc',   -- Loét áp lực
        'loet_tinh_mach',-- Loét tĩnh mạch
        'bong_do_2'      -- Bỏng độ II
    )),
    location TEXT,                          -- Vị trí vết thương trên cơ thể
    created_date DATE DEFAULT CURRENT_DATE,
    actual_healed_date DATE,                -- Ngày lành thật (điền khi lành)
    actual_days INTEGER,                    -- Số ngày lành thật (tính tự động)
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bảng 3: Mỗi lần thăm khám / thay băng
CREATE TABLE IF NOT EXISTS visits (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    wound_id UUID REFERENCES wounds(id) ON DELETE CASCADE,
    visit_date DATE DEFAULT CURRENT_DATE,
    length_cm FLOAT CHECK (length_cm > 0),
    width_cm FLOAT CHECK (width_cm > 0),
    depth_cm FLOAT CHECK (depth_cm >= 0),
    dressing_per_week INTEGER CHECK (dressing_per_week BETWEEN 1 AND 7),
    nurse_type TEXT CHECK (nurse_type IN ('specialist', 'general')),
    image_url TEXT,                         -- Đường dẫn ảnh trên Supabase Storage
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bảng 4: Log mỗi lần mô hình dự báo
CREATE TABLE IF NOT EXISTS predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    wound_id UUID REFERENCES wounds(id) ON DELETE CASCADE,
    visit_id UUID REFERENCES visits(id) ON DELETE SET NULL,
    predicted_days INTEGER,                 -- Mô hình dự báo bao nhiêu ngày
    confidence_low INTEGER,                 -- Khoảng tin cậy thấp
    confidence_high INTEGER,                -- Khoảng tin cậy cao
    model_version TEXT DEFAULT 'v1.0',
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- Tự động tính actual_days khi điền ngày lành
-- ============================================
CREATE OR REPLACE FUNCTION calc_actual_days()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_healed_date IS NOT NULL AND NEW.created_date IS NOT NULL THEN
        NEW.actual_days = NEW.actual_healed_date - NEW.created_date;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_calc_days
BEFORE INSERT OR UPDATE ON wounds
FOR EACH ROW EXECUTE FUNCTION calc_actual_days();

-- ============================================
-- Tạo Storage bucket để lưu ảnh vết thương
-- ============================================
INSERT INTO storage.buckets (id, name, public)
VALUES ('wound-images', 'wound-images', false)
ON CONFLICT (id) DO NOTHING;
