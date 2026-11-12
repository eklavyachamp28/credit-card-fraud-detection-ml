# Advanced ML & Statistical Modeling

Independent synthetic-data portfolio project demonstrating **XGBoost**, **Optuna** hyperparameter optimization, **SHAP** explainability, bootstrap confidence intervals, a paired permutation hypothesis test, and **MLflow** tracking.

Run: `pip install -r advanced-ml-statistical-modeling/requirements.txt` then `python advanced-ml-statistical-modeling/train.py`.

The script uses a stratified untouched holdout set, three-fold CV on development data, a logistic-regression baseline, PR-AUC/ROC-AUC, and optional SHAP explanations. Data and results are synthetic; no real-world performance is claimed. This subproject is housed within the fraud-detection repository pending a dedicated repository.
