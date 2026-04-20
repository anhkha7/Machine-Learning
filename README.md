# 🩺 Hệ thống Dự đoán Nguy cơ Tiểu đường (Diabetes Prediction System)

Dự án này xây dựng một ứng dụng web trực quan bằng **Streamlit** để dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số sức khỏe y tế cơ bản của người dùng. Hệ thống sử dụng mô hình học máy **Random Forest** được huấn luyện qua thư viện `scikit-learn` từ bộ dữ liệu `diabetes.csv`.

---

## 🎯 Mục tiêu dự án

- Cung cấp giao diện trực quan, thân thiện cho phép người dùng nhập nhanh các chỉ số sức khỏe của bản thân.
- Ứng dụng mô hình Machine Learning để tự động phân tích và đưa ra **cảnh báo sớm** hoặc **tin vui** về nguy cơ mắc bệnh tiểu đường.
- Cung cấp mã nguồn dễ hiểu, để dễ dàng quản lý việc huấn luyện lại mô hình nếu có tập dữ liệu mới.
- Lưu và tải trước mô hình (Serialization) để có thể dự đoán nhanh chóng trên Web App.

## ⚙️ Công nghệ sử dụng

- **Ngôn ngữ lập trình:** Python
- **Xây dựng Giao diện Web (UI):** [Streamlit](https://streamlit.io/)
- **Xử lý Hệ cơ sở dữ liệu:** Pandas, NumPy
- **Xây dựng & Đánh giá Mô hình:** Scikit-learn
- **Lưu trữ mô hình (Serialization):** Joblib

## 📁 Cấu trúc thư mục

```text
Machine-Learning/
├── app.py                   # Mã nguồn chính của ứng dụng Streamlit (Giao diện người dùng)
├── train_model.py           # Script tiền xử lý dữ liệu, huấn luyện Random Forest & đánh giá
├── diabetes.csv             # Tập dữ liệu đầu vào chứa hồ sơ bệnh án
├── diabetes_model.pkl       # File mô hình Random Forest đã được huấn luyện sẵn
├── diabetes_imputer.pkl     # File lưu thuật toán điền khuyết dữ liệu
└── README.md                # Tài liệu hiện tại hướng dẫn chi tiết dự án
```

## 📊 Mô tả Dữ liệu đầu vào

Mô hình yêu cầu 8 chỉ số sức khỏe đặc trưng:
1. **Pregnancies**: Số lần mang thai
2. **Glucose**: Chỉ số đường huyết (mg/dL)
3. **BloodPressure**: Huyết áp tâm trương (mm Hg)
4. **SkinThickness**: Độ dày nếp gấp da cơ tam đầu (mm)
5. **Insulin**: Chỉ số Insulin huyết thanh (mu U/ml)
6. **BMI**: Chỉ số khối cơ thể (Tính tỉ lệ cân nặng / chiều cao)
7. **DiabetesPedigreeFunction**: Chỉ số phả hệ tiểu đường (Tiền sử sức khỏe gia đình)
8. **Age**: Tuổi (năm)

**Cột mục tiêu (Outcome):**
- `1`: Cảnh báo nguy cơ mắc bệnh tiểu đường cao.
- `0`: Nguy cơ mắc bệnh thấp.

## 🛠 Quy trình Xử lý Dữ liệu & Huấn luyện (`train_model.py`)

Quy trình huấn luyện trong `train_model.py` diễn ra theo các bước chuẩn mực sau:
1. **Đọc dữ liệu** từ `diabetes.csv`.
2. **Tiền xử lý:** Nhận diện các giá trị `0` vô lý trong hồ sơ y tế ở những cột (như Glucose, BloodPressure, SkinThickness, Insulin, BMI) và thay thế thành giá trị rỗng `NaN`.
3. **Điền giá trị thiếu (Imputation):** Áp dụng `SimpleImputer` để tự động điền các số liệu bị khuyết bằng **trung vị (median)** của cột tương ứng.
4. **Chia tập dữ liệu:** Tách dữ liệu ra thành 2 tập Huấn luyện (Train) và Kiểm tra (Test) theo tỉ lệ `80/20`.
5. **Huấn luyện mô hình:** Sử dụng thuật toán `RandomForestClassifier` với 100 cây quyết định (`n_estimators=100`).
6. **Đánh giá & Báo cáo:** Tự động tính toán Độ chính xác (Accuracy), Ma trận nhầm lẫn (Confusion Matrix) và cung cấp Báo cáo chi tiết (Precision, Recall, F1-Score). Kết quả in ra Console rất hữu dụng nhằm sao chép vào file Báo cáo Đồ án.
7. **Lưu trữ:** Xuất kết quả mô hình (`diabetes_model.pkl`) và đối tượng imputer (`diabetes_imputer.pkl`) để lưu lại.

## 🚀 Hướng dẫn Cài đặt & Chạy ứng dụng

### 1. Yêu cầu môi trường
- Cài đặt Python (khuyến nghị phiên bản 3.9 trở lên).
- Khuyên dùng môi trường ảo (`Virtual Environment / venv`) để tránh xung đột cấu hình hệ thống.

### 2. Cài đặt các thư viện phụ thuộc

Mở Terminal (hoặc Command prompt) bằng quyền Admin trong thư mục dự án và cài đặt trực tiếp thông qua pip:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

> *(Tuỳ chọn)* Bạn có thể tạo file quản lý dependency bằng cách dùng lệnh `pip freeze > requirements.txt` sau khi quá trình cài đặt thành công.

### 3. Huấn luyện lại mô hình (Tùy chọn)

Nếu bạn muốn tạo lại file mô hình từ file dữ liệu chuyên biệt của mình hoặc để đồng bộ lại thông số, hãy chạy:

```bash
python train_model.py
```
> **Lưu ý:** Thao tác này sẽ tự động cập nhật hoặc ghi đè các tệp `.pkl` mới và hiển thị những chỉ số đánh giá lên màn hình Terminal.

### 4. Khởi động Web App

Khởi động giao diện Streamlit bằng lệnh:

```bash
streamlit run app.py 
hoặc
py -m streamlit run app.py
```

Trình duyệt sẽ khởi tạo giao diện sử dụng tại:
👉 **[http://localhost:8501](http://localhost:8501)**

## 💡 Cách sử dụng Web App

1. Tại giao diện nhập liệu sẽ hiển thị 8 thông số đầu vào. Hãy cung cấp thông tin y tế của bạn và tùy chỉnh cho từng thông số.
2. Nhấn nút **"🔍 Phân tích nguy cơ"** nằm ở phía dưới.
3. **Xem kết quả dự báo:**
   - ⚠️ **CẢNH BÁO (Hộp kiểm đỏ):** Cảnh báo hệ thống đánh giá bạn ở mức rủi ro CAO về tiểu đường. Khuyến nghị bạn đi kiểm tra y tế chi tiết.
   - ✅ **TIN VUI (Hộp kiểm xanh):** Nguy cơ tiểu đường THẤP theo chỉ số hiện tại. Nên duy trì chế độ sinh hoạt và theo sát chu kỳ sức khỏe.

## 📌 Lưu ý Quan trọng

- Phần mềm này xây dựng dựa trên mục đích **hỗ trợ học tập, minh họa quy trình làm Machine Learning**.
- Kết quả từ hệ thống AI này có ý nghĩa **tham khảo**. Tuyệt đối **không thay thế cho chẩn đoán chuyên khoa y tế từ bác sĩ**.
- Mô hình hiện tại cần được chạy với tệp `diabetes_model.pkl` ở cùng cấp thư mục dự án.

## 📈 Định hướng Phát triển trong Tương lai

- Bổ sung `requirements.txt` bằng câu lệnh để tiện thiết lập qua các môi trường khác nhau.
- Áp dụng các pipeline bài bản đầy đủ (tích hợp `imputer` cho dữ liệu mới ở đoạn suy luận `trên Web `app.py`).
- Cung cấp thêm biểu đồ trực quan như đồ thị đặc trưng, xác suất đưa ra (Predictive Probability) một cách số liệu nhất.
- Triển khai Hosting Public lên Hugging Face, Render hoặc Streamlit Community Cloud.

---

**Tác giả :**  
- Sinh viên 1: Tống Anh Kha - 123000367
- Sinh viên 2: Nguyễn Thanh Nam - 123000156
- Giáo viên hướng dẫn: Trần Văn Thành 

## 📝 Giấy phép (License)

Dự án này được phân phối dưới **Giấy phép MIT (MIT License)**. Bạn có thể tự do:
- **Sử dụng:** Cho mục đích thương mại hoặc cá nhân.
- **Sửa đổi & Phân phối:** Thay đổi và chia sẻ mã nguồn.

*Vui lòng giữ lại thông báo bản quyền và giấy phép khi phân phối mã nguồn.* Chi tiết xem tại [MIT License](https://opensource.org/licenses/MIT).
