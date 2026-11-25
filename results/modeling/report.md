# Kết quả dự đoán khách hàng rời mạng

Chọn **Random Forest** với bộ **full** (16 cột gốc). ROC-AUC 5-fold trên train đạt 0.8474 ± 0.0108 (độ lệch chuẩn).

## Lựa chọn trên tập huấn luyện

Trong nhóm đầy đủ, CatBoost dẫn đầu với AUC 0.8491. Trong nhóm rút gọn, CatBoost dẫn đầu với AUC 0.8445. Cấu hình có điểm trung bình cao nhất là CatBoost (full).
Mỗi tổ hợp giữ tham số có AUC trung bình cao nhất. Khi cách tổ hợp dẫn đầu tối đa 0,002 điểm trung bình và 0,005 ở từng fold, ưu tiên ít cột hơn, sau đó ưu tiên mô hình theo thứ tự Logistic Regression, CART, Random Forest, XGBoost, CatBoost.

| Mô hình | Bộ cột | AUC trung bình | Độ lệch chuẩn |
|---|---|---:|---:|
| CatBoost | full | 0.8491 | 0.0117 |
| XGBoost | full | 0.8489 | 0.0118 |
| Random Forest | full | 0.8474 | 0.0108 |
| Logistic Regression | full | 0.8466 | 0.0116 |
| CatBoost | reduced | 0.8445 | 0.0138 |
| XGBoost | reduced | 0.8439 | 0.0129 |
| Random Forest | reduced | 0.8434 | 0.0119 |
| Logistic Regression | reduced | 0.8374 | 0.0124 |
| CART | full | 0.8317 | 0.0080 |
| CART | reduced | 0.8304 | 0.0092 |

Tham số đã chọn `{"max_depth": 12, "min_samples_leaf": 20, "n_estimators": 100}`.

Thuộc tính đầu vào: tenure, MonthlyCharges, SeniorCitizen, Partner, Dependents, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod.

Ngưỡng 0.296409 tối đa hóa F1 của lớp rời mạng trên dự đoán ngoài fold (F1 0.6358, precision 0.5363, recall 0.7806).

## Đánh giá cuối

Trên 1409 khách test, ROC-AUC = 0.8440, precision = 0.5289, recall = 0.7834 và F1 = 0.6315.

| Thực tế | Dự đoán ở lại | Dự đoán rời mạng |
|---|---:|---:|
| Ở lại | 774 | 261 |
| Rời mạng | 81 | 293 |

Bỏ sót 81 khách rời mạng và cảnh báo nhầm 261 khách ở lại.

## Giải thích dự đoán

Đo đóng góp bằng giá trị SHAP, đơn vị xác suất. Trên mẫu cố định 256 khách train, các đặc trưng có đóng góp tuyệt đối trung bình lớn nhất là Contract_Month-to-month (0.074), InternetService_Fiber optic (0.047), tenure (0.043), tenure_group_0-12 (0.035), PaymentMethod_Electronic check (0.030).

Ví dụ lấy dòng đầu của tập test, khách `4376-KFVRS`. Hợp đồng Two year, thời gian sử dụng 72 tháng, cước tháng 114.05. Xác suất rời mạng 2.90%, nhãn dự đoán 0, nhãn thực tế 0.

Từ mức tham chiếu 0.2651 (xác suất), đóng góp dương làm tăng và đóng góp âm làm giảm điểm rời mạng của khách này. Năm đóng góp lớn nhất theo trị tuyệt đối:

| Đặc trưng | Đóng góp |
|---|---:|
| Contract_Month-to-month | -0.0780 |
| tenure | -0.0577 |
| Contract_Two year | -0.0484 |
| tenure_group_0-12 | -0.0306 |
| InternetService_Fiber optic | +0.0287 |
