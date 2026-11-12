"""Advanced synthetic-data modeling and statistical evaluation demo."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna


def bootstrap_ci(y, scores, n=300, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) == 2:
            values.append(average_precision_score(y[ids], scores[ids]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def permutation_p(y, a, b, repeats=1000, seed=42):
    rng = np.random.default_rng(seed)
    observed = average_precision_score(y, a) - average_precision_score(y, b)
    extreme = 0
    for _ in range(repeats):
        swap = rng.random(len(y)) < 0.5
        aa, bb = a.copy(), b.copy()
        aa[swap], bb[swap] = b[swap], a[swap]
        difference = average_precision_score(y, aa) - average_precision_score(y, bb)
        extreme += abs(difference) >= abs(observed)
    return (extreme + 1) / (repeats + 1)


def train(trials=12, seed=42, explain=True, track=True):
    X, y = make_classification(n_samples=3500, n_features=20, n_informative=10,
        n_redundant=4, weights=[0.94, 0.06], random_state=seed)
    dev_x, test_x, dev_y, test_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    imbalance = (len(dev_y) - dev_y.sum()) / max(1, dev_y.sum())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 80, 260),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0))
        scores = []
        for train_ids, val_ids in cv.split(dev_x, dev_y):
            model = XGBClassifier(**params, scale_pos_weight=imbalance,
                eval_metric="logloss", random_state=seed, n_jobs=1)
            model.fit(dev_x[train_ids], dev_y[train_ids])
            scores.append(average_precision_score(
                dev_y[val_ids], model.predict_proba(dev_x[val_ids])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    model = XGBClassifier(**study.best_params, scale_pos_weight=imbalance,
        eval_metric="logloss", random_state=seed, n_jobs=1)
    model.fit(dev_x, dev_y)
    baseline = make_pipeline(StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1500, random_state=seed))
    baseline.fit(dev_x, dev_y)
    a = model.predict_proba(test_x)[:, 1]
    b = baseline.predict_proba(test_x)[:, 1]
    metrics = dict(pr_auc=float(average_precision_score(test_y, a)),
        roc_auc=float(roc_auc_score(test_y, a)),
        baseline_pr_auc=float(average_precision_score(test_y, b)),
        pr_auc_ci95=bootstrap_ci(test_y, a),
        paired_permutation_p=float(permutation_p(test_y, a, b)),
        cv_pr_auc=float(study.best_value))
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if explain:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(shap.TreeExplainer(model).shap_values(test_x[:100]),
            test_x[:100], show=False)
        plt.savefig(out / "shap_summary.png", bbox_inches="tight")
        plt.close()
    if track:
        import mlflow
        mlflow.set_experiment("synthetic-advanced-ml")
        with mlflow.start_run():
            mlflow.log_params({"seed": seed, "trials": trials, **study.best_params})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_artifact(str(out / "metrics.json"))
            if explain:
                mlflow.log_artifact(str(out / "shap_summary.png"))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(trials=args.trials, explain=not args.no_shap,
        track=not args.no_mlflow), indent=2))
