import streamlit as st
import pandas as pd
import joblib

# Load mô hình đã lưu
model = joblib.load('diabetes_model.pkl')

# Cấu hình trang Web
st.set_page_config(page_title="Cảnh báo Tiểu đường", page_icon="🩺", layout="centered")

st.title('🩺 Hệ thống Dự đoán Nguy cơ Tiểu đường')
st.markdown('Vui lòng nhập các chỉ số sức khỏe dựa trên kết quả khám gần nhất của bạn.')
st.divider()

# Tạo 2 cột để giao diện nhập liệu đẹp hơn
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Số lần mang thai', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Chỉ số Glucose (mg/dL)', min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input('Huyết áp tâm trương (mm Hg)', min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input('Độ dày nếp gấp da (mm)', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input('Chỉ số Insulin (mu U/ml)', min_value=0, max_value=900, value=79)
    bmi = st.number_input('Chỉ số BMI', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input('Chỉ số phả hệ tiểu đường', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input('Tuổi', min_value=21, max_value=120, value=30)

st.divider()

# Xử lý khi nhấn nút Dự đoán
if st.button('🔍 Phân tích nguy cơ', type="primary", use_container_width=True):
    # Gom dữ liệu người dùng nhập thành bảng 1 dòng
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                             columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # AI dự đoán
    prediction = model.predict(user_data)
    
    # Hiển thị kết quả
    if prediction[0] == 1:
        st.error('⚠️ **CẢNH BÁO:** Hệ thống nhận thấy nguy cơ mắc bệnh tiểu đường CAO. Bạn nên đến cơ sở y tế để được bác sĩ kiểm tra chi tiết.')
    else:
        st.success('✅ **TIN VUI:** Nguy cơ mắc bệnh tiểu đường THẤP. Hãy tiếp tục duy trì chế độ ăn uống và sinh hoạt lành mạnh nhé!')