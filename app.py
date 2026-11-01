import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Cấu hình trang & Tải mô hình
st.set_page_config(page_title="Messi xG Predictor", page_icon="⚽", layout="wide")

# Dùng cache để server đám mây chỉ tải model 1 lần duy nhất, giúp web chạy mượt
@st.cache_resource
def load_model():
    return joblib.load('messi_xg_rf_full.pkl')

model_data = load_model()
model = model_data['model']
feature_cols = model_data['features']

# 2. Hàm hình học
def calculate_distance(x, y):
    return float(np.sqrt((120 - x)**2 + (40 - y)**2))

def calculate_angle(x, y):
    v1 = np.array([120 - x, 36 - y])
    v2 = np.array([120 - x, 44 - y])
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

# 3. Giao diện (Mặt tiền)
st.title("⚽ Trình dự đoán Bàn thắng của Lionel Messi (xG)")
st.markdown("Giả lập tình huống dứt điểm và xem AI đánh giá xác suất thành bàn dựa trên 17 năm dữ liệu.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Tọa độ & Không gian")
    x_coord = st.slider("Tọa độ X (Độ dọc sân)", min_value=60.0, max_value=120.0, value=105.0, step=0.5)
    y_coord = st.slider("Tọa độ Y (Độ ngang sân)", min_value=0.0, max_value=80.0, value=40.0, step=0.5)
    
    st.subheader("🛡️ Áp lực phòng ngự")
    defenders_in_cone = st.number_input("Số hậu vệ chắn trong phễu sút", min_value=0, max_value=11, value=1)
    close_defenders = st.number_input("Số hậu vệ áp sát (< 2m)", min_value=0, max_value=5, value=0)
    
    col_check1, col_check2 = st.columns(2)
    with col_check1:
        under_pressure = st.checkbox("Bị tì đè/áp lực mạnh")
    with col_check2:
        shot_first_time = st.checkbox("Sút một chạm (First time)")

with col2:
    st.subheader("⚙️ Bối cảnh & Kỹ thuật")
    play_pattern = st.selectbox("Tình huống bóng", ["Regular Play", "From Free Kick", "From Corner", "From Throw In", "From Counter", "Other"])
    shot_body_part = st.selectbox("Bộ phận dứt điểm", ["Left Foot", "Right Foot", "Head", "Other"])
    shot_type = st.selectbox("Loại cú sút", ["Open Play", "Free Kick", "Penalty"])
    shot_technique = st.selectbox("Kỹ thuật dứt điểm", ["Normal", "Half Volley", "Volley", "Lob", "Diving Header", "Backheel", "Overhead Kick"])

# 4. Xử lý tính toán (Backend thu nhỏ ẩn bên trong)
st.markdown("---")
if st.button("🚀 Dự đoán Xác suất Thành bàn (xG)", type="primary", use_container_width=True):
    dist = calculate_distance(x_coord, y_coord)
    ang = calculate_angle(x_coord, y_coord)
    
    # Tạo bảng dữ liệu gửi vào model
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    input_df['distance'] = dist
    input_df['angle'] = ang
    input_df['defenders_in_cone'] = defenders_in_cone
    input_df['close_defenders'] = close_defenders
    input_df['under_pressure'] = 1 if under_pressure else 0
    input_df['shot_first_time'] = 1 if shot_first_time else 0
    
    for prefix, val in [('play_pattern', play_pattern), ('shot_body_part', shot_body_part), 
                        ('shot_type', shot_type), ('shot_technique', shot_technique)]:
        col_name = f"{prefix}_{val}"
        if col_name in input_df.columns:
            input_df[col_name] = 1
            
    # Dự đoán
    xg_prob = float(model.predict_proba(input_df)[0][1])
    
    st.success("Phân tích hoàn tất!")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Xác suất thành bàn (xG)", f"{xg_prob * 100:.1f}%")
    with res_col2:
        st.metric("Khoảng cách tới gôn", f"{round(dist, 2)} m")
    with res_col3:
        st.metric("Góc mở khung thành", f"{round(ang, 2)}°")
        
    st.progress(xg_prob, text="Thước đo độ nguy hiểm")