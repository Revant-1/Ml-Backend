import io
from fastapi import FastAPI
from pydantic import BaseModel
import h5py
import joblib
import numpy as np
import pandas as pd

# Load Model & Scaler
def load_model(filename):
    """Load a Random Forest model and scaler from an .h5 file."""
    with h5py.File(filename, "r") as h5f:
        model_bytes = io.BytesIO(bytes(h5f["rf_model"][()]))  # Convert void to bytes
        scaler_bytes = io.BytesIO(bytes(h5f["scaler"][()]))

    model = joblib.load(model_bytes)
    scaler = joblib.load(scaler_bytes)

    print("✅ Model and Scaler loaded successfully")
    return model, scaler


rf_model, scaler = load_model("./rf_model.h5")

# FastAPI App
app = FastAPI()

# Define Input Data Model
class InputData(BaseModel):
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

# Prediction Endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict()])
    input_scaled = scaler.transform(input_df)

    probability = rf_model.predict_proba(input_scaled)[0][1]  # CVD probability (Class 1)

    # Convert to Risk Score (Log-Odds)
    risk_score = float('inf') if probability == 1 else float('-inf') if probability == 0 else np.log(probability / (1 - probability))

    return {
        "risk_score": round(risk_score, 2),
        "probability": round(probability * 100, 2)
    }

# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
