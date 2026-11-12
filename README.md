# Credit Card Fraud Detection (ML)

An end-to-end, reproducible baseline for detecting rare fraudulent transactions. The project uses a synthetic, imbalanced dataset so it runs without private financial data.

## What it demonstrates
- Data preparation and stratified train/test split
- Class-weighted logistic regression in a scikit-learn pipeline
- Imbalance-aware evaluation: precision, recall, F1, ROC-AUC and average precision (PR-AUC)
- Configurable decision threshold and saved model artifact
- FastAPI prediction endpoint with input validation

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs.

## Example
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount":125.50,"txn_hour":2,"distance_from_home_km":42.0,"velocity_1h":8,"is_foreign":1}'
```

The training script prints metrics and writes `model.joblib`. The API loads that artifact at startup; run training first. Synthetic results are illustrative, not evidence of real-world fraud performance. A production system would need temporal validation, leakage controls, calibrated thresholds, drift monitoring, explainability, access controls and review workflows.

## Stack
Python · pandas · NumPy · scikit-learn · joblib · FastAPI · pytest
