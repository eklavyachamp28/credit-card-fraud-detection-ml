"""FastAPI inference service for the trained fraud model."""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.joblib")
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    txn_hour: int = Field(ge=0, le=23)
    distance_from_home_km: float = Field(ge=0, le=100_000)
    velocity_1h: int = Field(ge=0, le=100_000)
    is_foreign: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(transaction: Transaction):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Run python train.py first.")
    model = load(MODEL_PATH)
    row = pd.DataFrame([transaction.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return {"fraud_probability": probability,
            "prediction": "review" if probability >= 0.5 else "normal",
            "threshold": 0.5}
