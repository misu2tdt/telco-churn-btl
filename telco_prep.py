"""telco_prep.py - Module tiền xử lý dùng chung cho BTL Data Mining (Telco Customer Churn).

Cách dùng (phần 2, 3, 4 của nhóm):

    import telco_prep as tp
    df = tp.load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = tp.split_data(df)      # 5634 train / 1409 test
    pipe = tp.with_model(LogisticRegression(max_iter=1000))   # mặc định: 28 đặc trưng (DROP_COLS)
    pipe.fit(X_train, y_train)

Mọi bước biến đổi nằm TRONG pipeline, nên khi cross-validation chỉ fit trên tập train
của từng fold (không rò rỉ dữ liệu). File này phải nằm cùng thư mục (hoặc trong
PYTHONPATH) khi nạp file .joblib, vì pipeline tham chiếu đến hàm prepare_features.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- hợp đồng dữ liệu
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"          # Yes/No trong file gốc -> 1/0 sau khi mã hóa
ID_COL = "customerID"

# 6 dịch vụ cộng thêm: giá trị gốc có 3 mức (Yes / No / No internet service)
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
# Các cột có mức "No phone service" / "No internet service" cần gộp về "No"
THREE_LEVEL_COLS = ["MultipleLines"] + ADDON_SERVICES

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "n_services"]
BINARY_PASSTHROUGH = ["SeniorCitizen"]          # đã là 0/1
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", *ADDON_SERVICES,
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
]

# Quyết định chọn đặc trưng của Phần 1 (xem báo cáo, mục Chọn đặc trưng): loại 3 cột
# -> 28 đặc trưng. Mặc định của build_pipeline / with_model; truyền drop_cols=[] để giữ đủ 31.
DROP_COLS = ["TotalCharges", "gender", "PhoneService"]

REDUCED_INPUTS = ["Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"]
MODEL_COLUMNS = [c for c in NUMERIC_COLS + BINARY_PASSTHROUGH + CATEGORICAL_COLS if c not in DROP_COLS]
FULL_INPUTS = [c for c in MODEL_COLUMNS if c not in ("n_services", "tenure_group")]

_TENURE_BINS = [-1, 12, 24, 48, np.inf]
_TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]


# ---------------------------------------------------------------- đọc / chia dữ liệu
def load_data(path: str) -> pd.DataFrame:
    """Đọc file CSV gốc (chưa xử lý gì)."""
    return pd.read_csv(path)


def encode_target(y: pd.Series) -> pd.Series:
    """Yes/No -> 1/0. Báo lỗi nếu có giá trị lạ."""
    out = y.map({"Yes": 1, "No": 0})
    if out.isna().any():
        raise ValueError("Cột Churn có giá trị khác Yes/No: "
                         f"{sorted(set(y[out.isna()].astype(str)))}")
    return out.astype(int)


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
    """Chia train/test (stratified theo Churn). X giữ nguyên cột gốc (kể cả customerID,
    pipeline sẽ tự bỏ cột này)."""
    y = encode_target(df[TARGET])
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


# ---------------------------------------------------------------- bước không học tham số
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch theo luật cố định (không học gì từ dữ liệu -> không gây rò rỉ).

    1. Cắt khoảng trắng thừa ở các cột chữ.
    2. Bỏ customerID (định danh, không mang thông tin dự đoán).
    3. TotalCharges: ép về số. Khách mới (tenure == 0) chưa phát sinh cước -> 0.
       Các ô trống khác (nếu có) ước lượng bằng MonthlyCharges * tenure.
    4. Gộp 'No phone service' / 'No internet service' về 'No' (thông tin này đã có
       ở cột PhoneService / InternetService, giữ lại chỉ tạo cột dư thừa).
    5. SeniorCitizen về số nguyên 0/1.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: v.strip() if isinstance(v, str) else v)

    out = out.drop(columns=[ID_COL], errors="ignore")

    if "TotalCharges" in out.columns:
        total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    else:  # ví dụ: dashboard không nhập TotalCharges
        total = pd.Series(np.nan, index=out.index)
    estimated = out["MonthlyCharges"].astype(float) * out["tenure"].astype(float)
    total = total.where(~(total.isna() & (out["tenure"] == 0)), 0.0)
    total = total.fillna(estimated)
    out["TotalCharges"] = total.astype(float)

    for c in THREE_LEVEL_COLS:
        if c in out.columns:
            out[c] = out[c].replace({"No internet service": "No",
                                     "No phone service": "No"})

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"]).astype(int)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo thuộc tính mới (Ch.2, mục 2.5 - attribute construction)."""
    out = df.copy()
    out["n_services"] = (out[ADDON_SERVICES] == "Yes").sum(axis=1).astype(int)
    out["tenure_group"] = pd.cut(out["tenure"], bins=_TENURE_BINS,
                                 labels=_TENURE_LABELS).astype(str)
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """clean_raw + add_features. Đây là hàm được gắn ở đầu pipeline."""
    return add_features(clean_raw(df))


def modeling_features(df: pd.DataFrame, reduced: bool) -> pd.DataFrame:
    """Project raw dashboard inputs to the same information for encoded and native models."""
    inputs = df.loc[:, REDUCED_INPUTS if reduced else FULL_INPUTS].copy()
    for c in ("tenure", "MonthlyCharges"):
        values = pd.to_numeric(inputs[c], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{c} phải là số hữu hạn không âm.")
        if c == "tenure" and (values % 1 != 0).any():
            raise ValueError("tenure phải là số tháng nguyên.")
        inputs[c] = values
    if "SeniorCitizen" in inputs and not pd.to_numeric(inputs["SeniorCitizen"]).isin([0, 1]).all():
        raise ValueError("SeniorCitizen phải là 0 hoặc 1.")
    features = clean_raw(inputs) if reduced else prepare_features(inputs)
    for c in inputs.columns.intersection(CATEGORICAL_COLS):
        if features[c].eq("").any():
            raise ValueError(f"{c} có giá trị trống.")
    return features.loc[:, REDUCED_INPUTS if reduced else MODEL_COLUMNS]



def predict_churn(bundle: dict, customers: pd.DataFrame) -> pd.DataFrame:
    """Score dashboard rows using the threshold selected on out-of-fold predictions."""
    inputs = customers.loc[:, bundle["input_columns"]]
    if inputs.isna().any().any():
        raise ValueError("Thiếu giá trị thuộc tính đầu vào.")
    probability = bundle["pipeline"].predict_proba(inputs)[:, 1]
    return pd.DataFrame({"probability": probability, "Churn": (probability >= bundle["threshold"]).astype(int)},
                        index=customers.index)


# ---------------------------------------------------------------- pipeline
def build_encoder(drop_cols: Iterable[str] | None = None) -> ColumnTransformer:
    """Chuẩn hóa z-score cho biến số, one-hot cho biến phân loại.
    Chỉ fit trên tập train (khi gọi pipe.fit(X_train, y_train))."""
    drop = set(DROP_COLS if drop_cols is None else drop_cols)
    num = [c for c in NUMERIC_COLS if c not in drop]
    binary = [c for c in BINARY_PASSTHROUGH if c not in drop]
    cat = [c for c in CATEGORICAL_COLS if c not in drop]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("bin", "passthrough", binary),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocessing_steps(drop_cols: Iterable[str] | None = None) -> list[tuple[str, object]]:
    """Danh sách bước (name, transformer) để ghép vào Pipeline của sklearn
    HOẶC imblearn.Pipeline (thêm SMOTE sau bước 'encode')."""
    return [
        ("prepare", FunctionTransformer(prepare_features, validate=False)),
        ("encode", build_encoder(drop_cols)),
    ]


def build_pipeline(drop_cols: Iterable[str] | None = None) -> Pipeline:
    """Pipeline chỉ tiền xử lý (chưa có mô hình)."""
    return Pipeline(preprocessing_steps(drop_cols))


def with_model(model, drop_cols: Iterable[str] | None = None) -> Pipeline:
    """Pipeline đầy đủ: tiền xử lý + mô hình."""
    return Pipeline([*preprocessing_steps(drop_cols), ("model", model)])


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Tên cột sau mã hóa của một pipeline đã fit."""
    return list(fitted_pipeline.named_steps["encode"].get_feature_names_out())