"""Opt-in MLflow / Weights & Biases experiment tracking.

Never log raw transaction records or secrets. Usage:
    python experiment_tracking.py --tracker mlflow
    python experiment_tracking.py --tracker wandb
"""
import argparse
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def run_experiment(tracker: str = "mlflow", seed: int = 42) -> dict:
    X, y = make_classification(n_samples=2000, n_features=12, n_informative=7,
                               weights=[0.97, 0.03], random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=seed)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {"roc_auc": float(roc_auc_score(y_test, scores)),
               "pr_auc": float(average_precision_score(y_test, scores))}
    config = {"seed": seed, "class_weight": "balanced", "samples": len(y)}
    if tracker == "mlflow":
        import mlflow
        mlflow.set_experiment("fraud-detection-synthetic")
        with mlflow.start_run():
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)
    elif tracker == "wandb":
        import wandb
        with wandb.init(project="fraud-detection-synthetic", config=config) as run:
            run.log(metrics)
    elif tracker != "none":
        raise ValueError("tracker must be mlflow, wandb, or none")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=["mlflow", "wandb", "none"], default="mlflow")
    args = parser.parse_args()
    print(run_experiment(tracker=args.tracker))
