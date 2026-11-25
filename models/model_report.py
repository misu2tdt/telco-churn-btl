"""Báo cáo lựa chọn mô hình và phân rã dự đoán của một khách hàng."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, DMatrix

import telco_prep as tp


def explain(pipe, train, example):
    model = pipe.named_steps["model"]
    sample = train.sample(min(len(train), 256), random_state=tp.RANDOM_STATE)
    transformed = pipe[:-1].transform(pd.concat([example, sample], ignore_index=True))
    names = list(transformed.columns) if isinstance(transformed, pd.DataFrame) else tp.feature_names(pipe)
    unit = "log-odds"
    coefficients = None
    if isinstance(model, LogisticRegression):
        center = pipe[:-1].transform(train).mean(axis=0)
        coefficients = model.coef_[0]
        values = (transformed - center) * coefficients
        baseline = model.intercept_[0] + center @ coefficients
    elif isinstance(model, CatBoostClassifier):
        contributions = model.get_feature_importance(
            Pool(transformed, cat_features=model.get_cat_feature_indices()), type="ShapValues")
        values, baseline = contributions[:, :-1], contributions[0, -1]
    elif isinstance(model, XGBClassifier):
        contributions = model.get_booster().predict(DMatrix(transformed), pred_contribs=True)
        values, baseline = contributions[:, :-1], contributions[0, -1]
    else:
        import shap

        explanation = shap.TreeExplainer(model)(transformed)
        values, baseline = explanation.values[:, :, 1], explanation.base_values[0, 1]
        unit = "xác suất"
    reconstructed = baseline + values[0].sum()
    if unit == "log-odds":
        reconstructed = 1 / (1 + np.exp(-reconstructed))
    np.testing.assert_allclose(reconstructed, pipe.predict_proba(example)[0, 1], atol=1e-5)
    table = pd.DataFrame({"feature": names, "mean_abs_contribution": np.abs(values[1:]).mean(axis=0),
                          "example_contribution": values[0]})
    if coefficients is not None:
        table["coefficient"] = coefficients
    return table, baseline, unit


def write_report(directory, bundle, winner, best, metrics, train, test, predictions, y_test):
    example = test.iloc[[0]]
    table, baseline, unit = explain(bundle["pipeline"], train, example)
    ranked = table.sort_values("mean_abs_contribution", ascending=False)
    ranked.to_csv(directory / "explanation.csv", index=False)
    local = table.sort_values("example_contribution", key=abs, ascending=False).head(5)
    leader = best.loc[best.mean_auc.idxmax()]
    reduced = best[best.feature_set == "reduced"].sort_values("mean_auc", ascending=False).iloc[0]
    full = best[best.feature_set == "full"].sort_values("mean_auc", ascending=False).iloc[0]
    test_result, oof = metrics.loc["test"], metrics.loc["train_oof"]
    threshold = bundle["threshold"]
    method = "hệ số nhân với độ lệch so với trung bình train" if "coefficient" in table else "giá trị SHAP"
    lines = [
        "# Kết quả dự đoán khách hàng rời mạng", "",
        f"Chọn **{winner.model}** với bộ **{winner.feature_set}** ({len(bundle['input_columns'])} cột gốc). "
        f"ROC-AUC 5-fold trên train đạt {winner.mean_auc:.4f} ± {winner.std_auc:.4f} (độ lệch chuẩn).",
        "", "## Lựa chọn trên tập huấn luyện", "",
        f"Trong nhóm đầy đủ, {full.model} dẫn đầu với AUC {full.mean_auc:.4f}. "
        f"Trong nhóm rút gọn, {reduced.model} dẫn đầu với AUC {reduced.mean_auc:.4f}. "
        f"Cấu hình có điểm trung bình cao nhất là {leader.model} ({leader.feature_set}).",
        "Mỗi tổ hợp giữ tham số có AUC trung bình cao nhất. Khi cách tổ hợp dẫn đầu tối đa 0,002 "
        "điểm trung bình và 0,005 ở từng fold, ưu tiên ít cột hơn, sau đó ưu tiên mô hình theo thứ tự "
        "Logistic Regression, CART, Random Forest, XGBoost, CatBoost.",
        "", "| Mô hình | Bộ cột | AUC trung bình | Độ lệch chuẩn |", "|---|---|---:|---:|",
    ]
    for row in best.sort_values("mean_auc", ascending=False).itertuples():
        lines.append(f"| {row.model} | {row.feature_set} | {row.mean_auc:.4f} | {row.std_auc:.4f} |")
    lines += [
        "", f"Tham số đã chọn `{winner.params}`.",
        "", "Thuộc tính đầu vào: " + ", ".join(bundle["input_columns"]) + ".",
        "", f"Ngưỡng {threshold:.6f} tối đa hóa F1 của lớp rời mạng trên dự đoán ngoài fold "
        f"(F1 {oof.f1:.4f}, precision {oof.precision:.4f}, recall {oof.recall:.4f}).",
        "", "## Đánh giá cuối", "",
        f"Trên {len(test)} khách test, ROC-AUC = {test_result.roc_auc:.4f}, precision = {test_result.precision:.4f}, "
        f"recall = {test_result.recall:.4f} và F1 = {test_result.f1:.4f}.",
        "", "| Thực tế | Dự đoán ở lại | Dự đoán rời mạng |", "|---|---:|---:|",
        f"| Ở lại | {int(test_result.tn)} | {int(test_result.fp)} |",
        f"| Rời mạng | {int(test_result.fn)} | {int(test_result.tp)} |",
        "", f"Bỏ sót {int(test_result.fn)} khách rời mạng và cảnh báo nhầm {int(test_result.fp)} khách ở lại.",
        "", "## Giải thích dự đoán", "",
        f"Đo đóng góp bằng {method}, đơn vị {unit}. Trên mẫu cố định {min(256, len(train))} khách train, "
        "các đặc trưng có đóng góp tuyệt đối trung bình lớn nhất là "
        + ", ".join(f"{r.feature} ({r.mean_abs_contribution:.3f})" for r in ranked.head(5).itertuples()) + ".",
        "", f"Ví dụ lấy dòng đầu của tập test, khách `{example.iloc[0][tp.ID_COL]}`. "
        f"Hợp đồng {example.iloc[0]['Contract']}, thời gian sử dụng {example.iloc[0]['tenure']} tháng, "
        f"cước tháng {example.iloc[0]['MonthlyCharges']}. Xác suất rời mạng {predictions.probability.iloc[0]:.2%}, "
        f"nhãn dự đoán {predictions.Churn.iloc[0]}, nhãn thực tế {y_test.iloc[0]}.",
        "", f"Từ mức tham chiếu {baseline:.4f} ({unit}), đóng góp dương làm tăng và đóng góp âm làm giảm "
        "điểm rời mạng của khách này. Năm đóng góp lớn nhất theo trị tuyệt đối:", "",
        "| Đặc trưng | Đóng góp |", "|---|---:|",
        *[f"| {r.feature} | {r.example_contribution:+.4f} |" for r in local.itertuples()],
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
