"""
test_connection.py
Chạy file này để kiểm tra kết nối Supabase thành công chưa
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Đọc thông tin kết nối từ file .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def test_connection():
    try:
        # Tạo kết nối
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Thử đọc dữ liệu từ bảng patients
        result = supabase.table("patients").select("*").limit(1).execute()

        print("✅ Kết nối Supabase thành công!")
        print(f"   URL: {SUPABASE_URL}")
        print(f"   Bảng patients: sẵn sàng")
        return True

    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return False

if __name__ == "__main__":
    test_connection()
