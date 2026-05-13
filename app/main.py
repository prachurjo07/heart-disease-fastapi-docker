import joblib
import pandas as pd

from fastapi import FastAPI
from app.schemas import HeartDiseaseInput


# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model/heart_model.joblib"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]
model_type = model_data["model_type"]
accuracy = model_data["accuracy"]


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="A FastAPI app for predicting heart disease using a trained ML classifier.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to the Heart Disease Prediction API",
        "docs": "Go to /docs to test the API",
        "health": "Go to /health to check server status",
        "info": "Go to /info to see model information"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Heart Disease Prediction API is running."
    }


# -----------------------------
# Info endpoint
# -----------------------------
@app.get("/info")
def model_info():
    return {
        "model_type": model_type,
        "features": features,
        "accuracy": round(float(accuracy), 4),
        "target": "heart disease presence or absence"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_heart_disease(input_data: HeartDiseaseInput):
    input_dict = input_data.model_dump()

    input_df = pd.DataFrame([input_dict], columns=features)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()

    return {
        "heart_disease": bool(prediction),
        "prediction": int(prediction),
        "probability_no_disease": round(probability[0], 4),
        "probability_disease": round(probability[1], 4)
    }