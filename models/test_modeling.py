"""Kiểm tra các đường dự đoán và giải thích chỉ bằng dữ liệu train."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import telco_prep as tp
from .model_report import explain
from .train_models import FOLDS, MODELS, choose_model, choose_threshold, make_pipeline


class ModelingTest(unittest.TestCase):
    def test_all_model_inputs_and_explanations(self):
        X = pd.read_csv("data/processed/X_train.csv", nrows=120)
        y = pd.read_csv("data/processed/y_train.csv", nrows=120)["Churn"]
        for name in MODELS:
            for feature_set, columns in [("reduced", tp.REDUCED_INPUTS), ("full", tp.FULL_INPUTS)]:
                with self.subTest(model=name, feature_set=feature_set):
                    params = {k: v[0] for k, v in MODELS[name][1].items()}
                    pipe = make_pipeline(name, feature_set, params).fit(X[columns], y)
                    one = X.iloc[[0]][columns]
                    probability = pipe.predict_proba(one)[0, 1]
                    bundle = dict(input_columns=columns, pipeline=pipe, threshold=probability)
                    self.assertEqual(tp.predict_churn(bundle, one).Churn.iloc[0], 1)
                    frame, _, _ = explain(pipe, X[columns], one)
                    expected = len(tp.REDUCED_INPUTS if feature_set == "reduced" else tp.MODEL_COLUMNS)
                    if name != "CatBoost":
                        expected = 12 if feature_set == "reduced" else 28
                    self.assertEqual(len(frame), expected)

    def test_threshold(self):
        y, p = [0, 1, 0, 1, 1], np.array([0.11, 0.34, 0.34, 0.95, 0.45])
        threshold = choose_threshold(y, p)
        self.assertEqual(f1_score(y, p >= threshold), max(f1_score(y, p >= t) for t in p))

    def test_dashboard_inputs(self):
        row = pd.read_csv("data/processed/X_train.csv", nrows=1)[tp.FULL_INPUTS]
        expected = tp.modeling_features(row, reduced=False)
        typed = row.astype({"tenure": str, "MonthlyCharges": str, "SeniorCitizen": str})
        pd.testing.assert_frame_equal(tp.modeling_features(typed, reduced=False), expected)
        for col, value in [("SeniorCitizen", 1.7), ("tenure", -2), ("tenure", 1.5),
                           ("MonthlyCharges", np.inf), ("Contract", "   ")]:
            with self.subTest(column=col, value=value), self.assertRaises(ValueError):
                tp.modeling_features(row.assign(**{col: value}), reduced=False)

    def test_close_scores_use_paired_folds(self):
        def row(model, features, scores):
            return dict(model=model, feature_set=features, mean_auc=np.mean(scores), **dict(zip(FOLDS, scores)))
        rows = pd.DataFrame([row("CatBoost", "full", [0.85] * 5),
                             row("CART", "reduced", [0.849] * 5),
                             row("Logistic Regression", "reduced", [0.83, 0.87, 0.85, 0.85, 0.845])])
        self.assertEqual(choose_model(rows).model, "CART")
        boundary = pd.DataFrame([row("CatBoost", "full", [0.85] * 5),
                                 row("CART", "reduced", [0.845, 0.845, 0.85, 0.85, 0.85])])
        self.assertEqual(choose_model(boundary).model, "CART")


if __name__ == "__main__":
    unittest.main()
