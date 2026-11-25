# Dự đoán khách hàng rời mạng (Telco Customer Churn)

Bài tập lớn môn Khai phá dữ liệu (CO3029), đề tài 01. Dữ liệu: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7043 khách, 21 thuộc tính, nhãn `Churn`).

| Phần | Nội dung               | Thư mục / file chính                                       |
|------|------------------------|------------------------------------------------------------|
| 1    | EDA và tiền xử lý      | `telco_prep.py`, `01_eda_tien_xu_ly.ipynb`, `make_data.py` |
| 2    | Mô hình phân loại      | `models/train_models.py`, `results/modeling/comparison.csv`       |
| 3    | Đánh giá và giải thích | `models/model_report.py`, `results/modeling/report.md`            |
| 4    | Dashboard Streamlit    | (nhóm thêm vào)                                            |

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Đặt file `WA_Fn-UseC_-Telco-Customer-Churn.csv` vào thư mục `data/` (nếu chưa có thì tải từ Kaggle ở trên).

## Cấu trúc thư mục

```
data/
  WA_Fn-UseC_-Telco-Customer-Churn.csv   dữ liệu gốc (không sửa)
  processed/                             X_train/X_test/y_train/y_test.csv (dữ liệu gốc đã chia, CHƯA mã hóa)
models/
  config.json                            quy ước chung (random_state, cột bị loại, tên 28 đặc trưng)
  preprocessor_fitted.joblib             pipeline tiền xử lý đã fit trên train (tạo bởi make_data.py, không commit)
image/                                   hình dùng cho báo cáo (cũng upload lên Overleaf)
telco_prep.py                            module tiền xử lý dùng chung
make_data.py                             tái tạo data/processed và models/ từ CSV gốc
01_eda_tien_xu_ly.ipynb                  EDA và tiền xử lý (Phần 1, có giải thích từng bước)
requirements.txt
```

## Cách dùng trong Phần 2, 3, 4

```python
import telco_prep as tp
from sklearn.linear_model import LogisticRegression

df = tp.load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X_train, X_test, y_train, y_test = tp.split_data(df)        # 5634 train / 1409 test, phân tầng, seed 42

pipe = tp.with_model(LogisticRegression(max_iter=1000, random_state=tp.RANDOM_STATE))     # tiền xử lý + mô hình, 28 đặc trưng
pipe.fit(X_train, y_train)                                  # mọi phép biến đổi chỉ fit trên train
proba = pipe.predict_proba(X_test)[:, 1]                    # xác suất churn
```

- Muốn lấy ma trận đã biến đổi (ví dụ cho SHAP): `pre = tp.build_pipeline().fit(X_train)`; `Xt = pre.transform(X_train)`; tên cột: `tp.feature_names(pre)`.
- Muốn thêm SMOTE: dùng `imblearn.pipeline.Pipeline([*tp.preprocessing_steps(), ("smote", SMOTE(random_state=tp.RANDOM_STATE)), ("model", ...)])`, SMOTE chỉ chạy khi `fit` nên không chạm vào tập kiểm tra.
- Dashboard nhận một khách mới dưới dạng DataFrame một dòng với các cột gốc (không cần `customerID`, `TotalCharges`, `Churn`).

## Quy ước chung (xin đọc trước khi làm)

1. **Không dùng tập kiểm tra để chọn mô hình hay chỉnh siêu tham số.** Dùng `StratifiedKFold` trên `X_train`; tập kiểm tra chỉ dùng một lần ở đánh giá cuối.
2. **Mất cân bằng lớp (26,54% churn) xử lý bên trong cross-validation** (`class_weight`, hoặc SMOTE trong pipeline), không resample trước khi chia.
3. **Không biến đổi dữ liệu ngoài pipeline.** Chuẩn hóa, mã hóa one-hot, tạo thuộc tính đều nằm trong `telco_prep.py`; đừng tự `fit_transform` trên toàn bộ dữ liệu.
4. **Cố định** `random_state=tp.RANDOM_STATE` **(42)** ở mọi nơi để kết quả tái lập.
5. Thước đo chính: AUC, recall, precision, F1 (không dùng accuracy vì một mô hình luôn đoán "ở lại" đã đạt 73,46%).
6. `telco_prep.py` phải nằm cùng thư mục chạy (hoặc trong `PYTHONPATH`) khi nạp file `.joblib`.

## Chạy thí nghiệm phân loại

Sau khi cài các thư viện, chạy từ thư mục gốc:

```bash
python -m models.train_models
```

Script tái tạo đúng tập chia của Phần 1 từ CSV gốc, dùng chung 5 fold phân tầng cho 64 cấu hình của 5 mô hình trên 2 bộ đầu vào. Mỗi tổ hợp giữ cấu hình có ROC-AUC trung bình cao nhất. Tiêu chí ưu tiên bộ nhỏ hoặc mô hình đơn giản khi điểm sát nhau được ghi trong báo cáo. Ngưỡng tối đa hóa F1 được chọn trên dự đoán ngoài fold của cấu hình cuối. Test chỉ dùng sau khi chốt các lựa chọn này.

- [Báo cáo thực nghiệm](results/modeling/report.md) gồm so sánh, ma trận nhầm lẫn và giải thích một khách hàng.
- [Bảng 10 tổ hợp](results/modeling/comparison.csv) và [toàn bộ cấu hình](results/modeling/cv_trials.csv) ghi tham số, AUC từng fold, trung bình và độ lệch chuẩn.
- [Chỉ số](results/modeling/metrics.csv), [đóng góp thuộc tính](results/modeling/explanation.csv) và [luật của cây CART](results/modeling/cart_rules.txt). Ngưỡng số trong luật CART ở thang chuẩn hóa của pipeline.
- `models/churn.joblib` lưu pipeline, danh sách cột gốc, tên mô hình, bộ thuộc tính và ngưỡng. File được tạo lại bằng lệnh huấn luyện.

Bản thực nghiệm đi kèm dùng Python 3.12.10, pandas 3.1.1, NumPy 2.5.5, scikit-learn 1.9.1, XGBoost 3.4.1, CatBoost 1.2.10, SHAP 0.54.1 và joblib 1.6.0. Kiểm tra cả hai bộ đầu vào, ngưỡng và các phương pháp giải thích bằng `python -m unittest -v models.test_modeling`.

Dashboard có thể truyền một dòng từ biểu mẫu dưới dạng DataFrame. Cột `Churn` đầu ra là nhãn 0/1 theo ngưỡng đã lưu.

```python
import joblib
import pandas as pd
from telco_prep import predict_churn

bundle = joblib.load("models/churn.joblib")
customer = pd.read_csv("data/processed/X_test.csv", nrows=1)[bundle["input_columns"]]
prediction = predict_churn(bundle, customer)
```

## Kết quả của Phần 1

- 7043 khách, chia 5634 train / 1409 test, tỉ lệ churn 26,54% ở cả hai tập.
- Làm sạch: `TotalCharges` ép về số (11 ô trống của khách `tenure = 0` điền 0), gộp nhãn "No internet/phone service" về "No" (9838 giá trị), bỏ `customerID`.
- Thuộc tính mới: `n_services`, `tenure_group`. Bỏ `TotalCharges`, `gender`, `PhoneService` (xem `tp.DROP_COLS`).
- Đầu ra: **28 đặc trưng**, không còn giá trị thiếu. AUC 5-fold CV với Logistic Regression khoảng 0,847 (chỉ để kiểm tra kỹ thuật, không phải kết quả mô hình chính thức).

## Tái tạo file bàn giao

```bash
python make_data.py
```

Script đọc CSV gốc, chia, fit pipeline, xuất `data/processed/` và `models/`, rồi tự kiểm tra (không NaN, 28 đặc trưng, nạp lại từ đĩa khớp, dòng kiểu dashboard chạy được).

## Khai báo sử dụng AI

TODO (nhóm điền): công cụ AI đã dùng, ở phần nào, và cách kiểm chứng kết quả.