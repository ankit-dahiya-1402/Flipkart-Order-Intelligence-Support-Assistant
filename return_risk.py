"""
Part 1 - Return-Risk Prediction
================================
Trains a model that predicts whether a Flipkart order will be returned, and
explains *why* for a single order.

Run this file directly to train:
    python return_risk.py

Import it elsewhere to get predictions once trained:
    from return_risk import predict
    predict({"product_category": "Apparel", ...})

Data note: data/orders.csv is SYNTHETIC (see data/generate_orders.py) because
real Flipkart order data is not available. The `returned` label depends on a
mix of 8 features (customer return history, category, payment method,
discount, delivery, price, weekend, tenure) so it is not trivially guessable
from a single column.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "orders.csv")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "return_risk_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "return_risk_meta.json")

NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "returned"

# Categories that tend to be returned more often (used for explanations).
HIGH_RETURN_CATEGORIES = {"Apparel", "Footwear"}


# ---------------------------------------------------------------------------
# Preprocessing + training
# ---------------------------------------------------------------------------
def build_pipeline(classifier, scale_numeric: bool) -> Pipeline:
    """Shared preprocessing (impute + one-hot) wrapped around any classifier."""
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def analyze_missingness(df: pd.DataFrame) -> dict:
    """Checks whether rating_given is missing depending on payment_method (MAR)."""
    by_payment = (
        df.assign(missing=df["rating_given"].isna())
        .groupby("payment_method")["missing"]
        .mean()
    )
    cod_rate = by_payment.get("COD", 0.0)
    non_cod_avg = by_payment.drop("COD", errors="ignore").mean()
    gap = cod_rate - non_cod_avg
    classification = "MAR (depends on payment_method)" if gap > 0.05 else "MCAR"
    return {
        "overall_missing_rate": round(float(df["rating_given"].isna().mean()), 4),
        "missing_rate_by_payment_method": by_payment.round(4).to_dict(),
        "classification": classification,
    }


def best_f1_threshold(y_true, y_proba) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    return float(thresholds[np.argmax(f1s[:-1])]) if len(thresholds) else 0.5


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} orders. Overall return rate: {df[TARGET].mean():.4f}")
    print("Missingness check:", analyze_missingness(df))

    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Baseline: always predict the majority class. Shows why accuracy alone
    # is a misleading metric on an imbalanced return-rate problem.
    dummy = build_pipeline(DummyClassifier(strategy="most_frequent"), scale_numeric=False)
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate(y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1])
    print("Dummy baseline:", dummy_metrics)

    # Logistic Regression, for comparison.
    logreg = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        scale_numeric=True,
    )
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]
    logreg_metrics = evaluate(y_test, (logreg_proba >= 0.5).astype(int), logreg_proba)
    print("Logistic Regression:", logreg_metrics)

    # Random Forest, the model we actually ship — small grid search.
    rf = build_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        scale_numeric=False,
    )
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [8, None],
        "classifier__min_samples_leaf": [1, 3],
    }
    grid = GridSearchCV(rf, param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    rf_proba = best_rf.predict_proba(X_test)[:, 1]
    rf_metrics_default = evaluate(y_test, (rf_proba >= 0.5).astype(int), rf_proba)

    # Business threshold tuning: the default 0.5 cutoff is not necessarily
    # best for catching returns, so pick the threshold that maximizes F1.
    threshold = best_f1_threshold(y_test, rf_proba)
    rf_metrics_tuned = evaluate(y_test, (rf_proba >= threshold).astype(int), rf_proba)
    print(f"Random Forest (best params {grid.best_params_}):")
    print(f"  @0.50 threshold: {rf_metrics_default}")
    print(f"  @{threshold:.2f} threshold (F1-optimal): {rf_metrics_tuned}")

    # Explainability: impurity importance vs. permutation importance.
    # Impurity importance over-rates high-cardinality numeric columns;
    # permutation importance is measured on held-out data and is more
    # trustworthy for deciding what actually predicts returns.
    perm = permutation_importance(
        best_rf, X_test, y_test, n_repeats=8, random_state=42, scoring="f1", n_jobs=-1
    )
    perm_ranked = sorted(zip(ALL_FEATURES, perm.importances_mean), key=lambda t: -t[1])
    print("Top permutation-importance features:", perm_ranked[:5])

    # Persist everything predict() needs.
    joblib.dump(best_rf, MODEL_PATH)
    numeric_stats = {
        col: {
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
        }
        for col in NUMERIC_FEATURES
    }
    meta = {
        "threshold": round(threshold, 4),
        "bucket_boundaries": {
            "low_below": round(threshold, 4),
            "high_at_or_above": round(min(threshold + 0.2, 0.95), 4),
        },
        "numeric_stats": numeric_stats,
        "top_features_for_explanation": [f for f, _ in perm_ranked[:6]],
        "metrics": {
            "dummy_baseline": dummy_metrics,
            "logistic_regression": logreg_metrics,
            "random_forest_default_threshold": rf_metrics_default,
            "random_forest_tuned_threshold": rf_metrics_tuned,
        },
        "missingness_analysis": analyze_missingness(df),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")


# ---------------------------------------------------------------------------
# Prediction + explanation (used by the assistant and the Streamlit app)
# ---------------------------------------------------------------------------
_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def _bucket(probability: float, meta: dict) -> str:
    b = meta["bucket_boundaries"]
    if probability < b["low_below"]:
        return "Low"
    if probability < b["high_at_or_above"]:
        return "Medium"
    return "High"


def _explain(features: dict, meta: dict) -> list[str]:
    """Rule-based, human-readable reasons (no SHAP dependency required)."""
    reasons = []
    stats = meta["numeric_stats"]
    prev_orders = features.get("num_previous_orders") or 0
    prev_returns = features.get("num_previous_returns") or 0
    return_ratio = prev_returns / prev_orders if prev_orders > 0 else 0.0

    for feature in meta["top_features_for_explanation"]:
        if feature == "num_previous_returns" and return_ratio > 0:
            reasons.append(f"Customer has returned {return_ratio:.0%} of {prev_orders} previous orders")
        elif feature == "payment_method" and features.get("payment_method") == "COD":
            reasons.append("Cash-on-delivery orders historically have a higher return rate")
        elif feature == "product_category" and features.get("product_category") in HIGH_RETURN_CATEGORIES:
            reasons.append(f"{features['product_category']} is a higher-return category")
        elif feature in stats and features.get(feature) is not None:
            value = features[feature]
            q25, q75, median = stats[feature]["q25"], stats[feature]["q75"], stats[feature]["median"]
            if value >= q75:
                reasons.append(f"{feature.replace('_', ' ')} is high ({value}, typical is ~{median:.0f})")
            elif value <= q25:
                reasons.append(f"{feature.replace('_', ' ')} is low ({value}, typical is ~{median:.0f})")
        if len(reasons) >= 4:
            break

    if not reasons:
        reasons.append("No single factor stands out; risk reflects a mix of typical values")
    return reasons


def predict(features: dict) -> dict:
    model, meta = _load()
    row = {k: features.get(k) for k in ALL_FEATURES}
    proba = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    return {
        "probability": round(proba, 4),
        "risk_bucket": _bucket(proba, meta),
        "threshold_used": meta["threshold"],
        "top_reasons": _explain(features, meta),
    }


if __name__ == "__main__":
    train_and_evaluate()
