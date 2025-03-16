import io
from fastapi import FastAPI
from pydantic import BaseModel
import h5py
import joblib
import numpy as np
import pandas as pd

# Function to Load Models & Scalers from .h5
def load_model(filename, model_key):
    """Load a model and scaler from an .h5 file."""
    with h5py.File(filename, "r") as h5f:
        model_bytes = io.BytesIO(bytes(h5f[model_key][()]))  # Convert void to bytes
        scaler_bytes = io.BytesIO(bytes(h5f["scaler"][()]))

    model = joblib.load(model_bytes)
    scaler = joblib.load(scaler_bytes)

    print(f"✅ {model_key} Model and Scaler loaded successfully")
    return model, scaler

# Load both models
rf_model, rf_scaler = load_model("rf_model.h5", "rf_model")
xgb_model, xgb_scaler = load_model("xgb_model.h5", "xgb_model")

# FastAPI App
app = FastAPI()

# ✅ Define Input Schema for Random Forest (rf_model)
class RFInput(BaseModel):
    age: int
    gender: int
    chestpain: int
    restingBP: int
    serumcholestrol: int
    fastingbloodsugar: int
    restingrelectro: int
    maxheartrate: int
    exerciseangia: int
    oldpeak: float
    noofmajorvessels: int

# ✅ Define Input Schema for XGBoost (xgb_model)
class XGBInput(BaseModel):
    age: int
    gender: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    ca: int

# ✅ Prediction function
def get_prediction(model, scaler, input_data):
    input_df = pd.DataFrame([input_data.dict()])
    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]  # Probability of CVD (Class 1)

    # Convert to Risk Score (Log-Odds)
    risk_score = float('inf') if probability == 1 else float('-inf') if probability == 0 else np.log(probability / (1 - probability))

    return {
        "risk_score": round(risk_score, 2),
        "probability": round(probability * 100, 2)
    }

# ✅ Prediction route for Random Forest
@app.post("/predict/rf")
async def predict_rf(input_data: RFInput):
    return get_prediction(rf_model, rf_scaler, input_data)

# ✅ Prediction route for XGBoost
@app.post("/predict/xgb")
async def predict_xgb(input_data: XGBInput):
    return get_prediction(xgb_model, xgb_scaler, input_data)

# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
