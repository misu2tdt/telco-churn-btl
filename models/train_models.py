"""So sánh 5 mô hình dự đoán churn, chọn ngưỡng trên train và đánh giá trên holdout."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

import telco_prep as tp
from .model_report import write_report

DATA = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
RESULTS = Path("results/modeling")
ARTIFACT = Path("models/churn.joblib")
FOLDS = [f"fold_{i}" for i in range(1, 6)]
SEED = tp.RANDOM_STATE
MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=SEED),
                            {"C": [0.1, 1.0], "class_weight": [None, "balanced"]}),
    "CART": (DecisionTreeClassifier(criterion="gini", random_state=SEED),
             {"max_depth": [3, 6], "min_samples_leaf": [15, 40]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=2),
                      {"n_estimators": [100, 200], "max_depth": [6, 12], "min_samples_leaf": [5, 20]}),
    "XGBoost": (XGBClassifier(random_state=SEED, n_jobs=2, tree_method="hist", eval_metric="logloss"),
                {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1], "max_depth": [2, 4]}),
    "CatBoost": (CatBoostClassifier(random_seed=SEED, thread_count=3, verbose=False,
                                    allow_writing_files=False),
                 {"iterations": [100, 200], "learning_rate": [0.03, 0.1], "depth": [3, 5]}),
}


def make_pipeline(name, feature_set, params):
    reduced = feature_set == "reduced"
    columns = tp.REDUCED_INPUTS if reduced else tp.MODEL_COLUMNS
    model = clone(MODELS[name][0]).set_params(**params)
    excluded = set(tp.NUMERIC_COLS + tp.CATEGORICAL_COLS + tp.BINARY_PASSTHROUGH) - set(columns)
    pipe = tp.with_model(model, drop_cols=excluded)
    pipe.set_params(prepare=FunctionTransformer(tp.modeling_features, kw_args={"reduced": reduced}))
    if name == "CatBoost":
        pipe.set_params(encode="passthrough", model__cat_features=[c for c in columns if c in tp.CATEGORICAL_COLS])
    return pipe


def compare(X, y):
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    trials, best_rows, probabilities = [], [], {}
    for name, (_, grid) in MODELS.items():
        for feature_set in ("full", "reduced"):
            best_auc = -np.inf
            for params in ParameterGrid(grid):
                scores, oof = [], np.full(len(y), np.nan)
                for train, valid in splits:
                    pipe = make_pipeline(name, feature_set, params)
                    pipe.fit(X.iloc[train], y.iloc[train])
                    oof[valid] = pipe.predict_proba(X.iloc[valid])[:, 1]
                    scores.append(roc_auc_score(y.iloc[valid], oof[valid]))
                assert np.isfinite(oof).all()
                row = dict(model=name, feature_set=feature_set, params=json.dumps(params),
                           mean_auc=np.mean(scores), std_auc=np.std(scores), **dict(zip(FOLDS, scores)))
                trials.append(row)
                if row["mean_auc"] > best_auc:
                    best_auc, best_row, best_oof = row["mean_auc"], row, oof
            best_rows.append(best_row)
            probabilities[name, feature_set] = best_oof
            print(f"{name:20} {feature_set:7} CV AUC {best_auc:.5f}", flush=True)
    return pd.DataFrame(trials), pd.DataFrame(best_rows), probabilities


def choose_model(best):
    leader = best.loc[best.mean_auc.idxmax()]
    # Predeclared practical tie: compare the same folds before preferring fewer inputs.
    paired_gap = best[FOLDS].sub(leader[FOLDS].astype(float)).abs().max(axis=1)
    # Round subtraction noise at the inclusive AUC cutoffs.
    mean_gap = (leader.mean_auc - best.mean_auc).round(14)
    close = best[(mean_gap <= 0.002) & (paired_gap.round(14) <= 0.005)].copy()
    close["inputs"] = close.feature_set.map({"full": len(tp.FULL_INPUTS), "reduced": len(tp.REDUCED_INPUTS)})
    close["complexity"] = close.model.map({name: i for i, name in enumerate(MODELS)})
    return close.sort_values(["inputs", "complexity", "mean_auc"], ascending=[True, True, False]).iloc[0]


def choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)
    return float(thresholds[np.argmax(f1)])


def metrics(y, probability, threshold):
    label = probability >= threshold
    tn, fp, fn, tp_count = confusion_matrix(y, label, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y, probability), precision=precision_score(y, label, zero_division=0),
                recall=recall_score(y, label), f1=f1_score(y, label),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp_count))


def main():
    X, test, y, y_test = tp.split_data(tp.load_data(DATA))
    trials, best, oof = compare(X, y)
    winner = choose_model(best)
    probability = oof[winner.model, winner.feature_set]
    threshold = choose_threshold(y, probability)
    pipe = make_pipeline(winner.model, winner.feature_set, json.loads(winner.params)).fit(X, y)
    inputs = tp.REDUCED_INPUTS if winner.feature_set == "reduced" else tp.FULL_INPUTS
    bundle = dict(pipeline=pipe, threshold=threshold, input_columns=inputs, model=winner.model,
                  feature_set=winner.feature_set)

    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ARTIFACT.parent) as stage:
        candidate = Path(stage) / ARTIFACT.name
        joblib.dump(bundle, candidate, compress=3)
        loaded = joblib.load(candidate)
        example = X.loc[:, inputs].iloc[[0]]
        pd.testing.assert_frame_equal(tp.predict_churn(loaded, example), tp.predict_churn(bundle, example))
        candidate.replace(ARTIFACT)

    test_scores = tp.predict_churn(loaded, test)
    evaluation = pd.DataFrame([metrics(y, probability, threshold),
                               metrics(y_test, test_scores.probability, threshold)], index=["train_oof", "test"])
    trials.to_csv(RESULTS / "cv_trials.csv", index=False)
    best.sort_values("mean_auc", ascending=False).to_csv(RESULTS / "comparison.csv", index=False)
    evaluation.to_csv(RESULTS / "metrics.csv", index_label="split")
    cart_row = best[(best.model == "CART") & (best.feature_set == "full")].iloc[0]
    cart = make_pipeline("CART", "full", json.loads(cart_row.params)).fit(X, y)
    rules = export_text(cart.named_steps["model"], feature_names=tp.feature_names(cart), show_weights=True)
    (RESULTS / "cart_rules.txt").write_text(rules, encoding="utf-8")
    write_report(RESULTS, bundle, winner, best, evaluation, X, test, test_scores, y_test)
    print(f"Selected {winner.model} ({winner.feature_set}); threshold={threshold:.6f}")
    print(evaluation.to_string())


if __name__ == "__main__":
    main()
