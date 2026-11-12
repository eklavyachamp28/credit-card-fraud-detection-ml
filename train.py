"""Train an imbalanced fraud-detection baseline on reproducible synthetic data."""
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["amount", "txn_hour", "distance_from_home_km", "velocity_1h", "is_foreign"]

def make_dataset(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.2, 1.0, n)
    hour = rng.integers(0, 24, n)
    distance = rng.exponential(18, n)
    velocity = rng.poisson(2, n)
    foreign = rng.binomial(1, 0.08, n)
    # Synthetic signal; intentionally rare and suitable only for a demo.
    score = (-5.2 + 0.012 * np.minimum(amount, 500)
             + 0.055 * distance + 0.38 * velocity
             + 1.15 * foreign + 0.75 * ((hour <= 4) | (hour >= 23)))
    probability = 1 / (1 + np.exp(-score))
    fraud = rng.binomial(1, np.clip(probability, 0, 0.85))
    return pd.DataFrame({"amount": amount, "txn_hour": hour,
                         "distance_from_home_km": distance,
                         "velocity_1h": velocity, "is_foreign": foreign,
                         "is_fraud": fraud})

def build_pipeline():
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), ["amount", "txn_hour", "distance_from_home_km", "velocity_1h"]),
        ("binary", "passthrough", ["is_foreign"]),
    ])
    return Pipeline([("preprocess", prep),
                     ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000))])

def train(output_path="model.joblib"):
    data = make_dataset()
    X, y = data[FEATURES], data["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model = build_pipeline().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, probabilities):.4f}")
    dump(model, output_path)
    return model

if __name__ == "__main__":
    train()
