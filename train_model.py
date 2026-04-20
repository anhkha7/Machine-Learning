import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib

print("1. Đang đọc dữ liệu...")
df = pd.read_csv('diabetes.csv')

print("2. Đang tiền xử lý dữ liệu (Xử lý các số 0 phi lý)...")
# Các cột không được phép có giá trị 0
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Khởi tạo bộ điền dữ liệu bằng giá trị trung vị (median)
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

print("3. Đang huấn luyện mô hình Random Forest...")
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá nhanh mô hình
accuracy = model.score(X_test, y_test)
print(f"-> Độ chính xác trên tập Test: {accuracy * 100:.2f}%")

# --- ĐOẠN CODE IN BÁO CÁO CHI TIẾT ĐÃ ĐƯỢC THÊM VÀO ĐÂY ---
print("\n--- BÁO CÁO CHI TIẾT (Dùng cho file Word) ---")
y_pred = model.predict(X_test)
print("1. Ma trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\n2. Các chỉ số Precision, Recall, F1-Score:")
print(classification_report(y_test, y_pred))
print("--------------------------------------------\n")

print("4. Đang lưu mô hình...")
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(imputer, 'diabetes_imputer.pkl')
print("✅ HOÀN TẤT! Đã tạo ra các file .pkl sẵn sàng cho Web App.")