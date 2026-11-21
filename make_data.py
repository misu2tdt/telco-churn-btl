"""make_data.py - Tái tạo toàn bộ file bàn giao của Phần 1 từ CSV gốc (không cần mở notebook).

    python make_data.py
    python make_data.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Tạo ra:
    data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv   (dữ liệu gốc đã chia, CHƯA mã hóa)
    models/preprocessor_fitted.joblib                                 (pipeline tiền xử lý, fit trên train)
    models/config.json                                                (quy ước chung của nhóm)

Cùng random_state nên mọi lần chạy cho cùng một cách chia. Script dừng với lỗi nếu có kiểm tra thất bại.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import telco_prep as tp

DEFAULT_DATA = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROC_DIR, MODEL_DIR = Path("data/processed"), Path("models")


def main(data_path: Path) -> None:
    if not data_path.exists():
        raise SystemExit(f"Không thấy {data_path}. Đặt file CSV gốc vào thư mục data/ hoặc dùng --data.")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. đọc và chia (chia TRƯỚC mọi phép biến đổi có học tham số)
    raw = tp.load_data(str(data_path))
    X_train, X_test, y_train, y_test = tp.split_data(raw)
    assert (len(X_train), len(X_test)) == (5634, 1409), "Số dòng train/test khác dự kiến (5634/1409)"

    # 2. pipeline tiền xử lý, fit CHỈ trên train
    pipe = tp.build_pipeline()                       # mặc định bỏ tp.DROP_COLS
    Xt_train = pipe.fit_transform(X_train)
    Xt_test = pipe.transform(X_test)
    names = tp.feature_names(pipe)
    assert not np.isnan(Xt_train).any() and not np.isnan(Xt_test).any(), "Còn NaN sau xử lý"
    assert Xt_train.shape[1] == Xt_test.shape[1] == len(names) == 28, "Số đặc trưng khác 28"

    # 3. xuất file
    X_train.to_csv(PROC_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROC_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROC_DIR / "y_test.csv", index=False)
    joblib.dump(pipe, MODEL_DIR / "preprocessor_fitted.joblib")
    config = {
        "target": tp.TARGET,
        "random_state": tp.RANDOM_STATE,
        "test_size": tp.TEST_SIZE,
        "drop_cols": tp.DROP_COLS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features_after_encoding": len(names),
        "feature_names": names,
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. kiểm tra: nạp lại từ đĩa, dòng kiểu dashboard, tỉ lệ lớp, CV AUC
    loaded = joblib.load(MODEL_DIR / "preprocessor_fitted.joblib")
    assert np.allclose(loaded.transform(X_test), Xt_test), "Nạp lại từ đĩa cho kết quả khác"
    one_row = X_test.iloc[[0]].drop(columns=["customerID", "TotalCharges"])
    assert loaded.transform(one_row).shape == (1, 28), "Dòng kiểu dashboard (thiếu ID, TotalCharges) lỗi"
    print(f"Tỉ lệ churn train/test: {y_train.mean():.4f} / {y_test.mean():.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=tp.RANDOM_STATE)
    auc = cross_val_score(tp.with_model(LogisticRegression(max_iter=1000)), X_train, y_train,
                          cv=cv, scoring="roc_auc")
    print(f"AUC 5-fold CV (Logistic Regression, chỉ để kiểm tra kỹ thuật): {auc.mean():.4f}")

    print("Đã xuất:", sorted(p.as_posix() for p in list(PROC_DIR.iterdir()) + list(MODEL_DIR.iterdir())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="đường dẫn CSV gốc")
    main(ap.parse_args().data)