from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
import boto3
import json
import os

app = FastAPI()

# Configuration
MINIO_ENDPOINT = "http://minio:9000"
BUCKET_NAME = "stock-data"
TICKER = "NVDA"
MODEL_KEY = f"models/production_model_{TICKER}.json"

# Connect to MinIO
s3 = boto3.client('s3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id='minio_admin',
    aws_secret_access_key='minio_password'
)

# Global variable to hold the model
model = None

@app.on_event("startup")
def load_model():
    global model
    print("📥 Downloading model from Data Lake...")
    # Download the JSON file locally
    s3.download_file(BUCKET_NAME, MODEL_KEY, "latest_model.json")
    
    # Load into XGBoost
    model = xgb.XGBRegressor()
    model.load_model("latest_model.json")
    print("✅ Model loaded and ready to serve!")

@app.post("/predict")
def predict(features: dict):
    """
    Expects a JSON payload like:
    { "Open": 130, "High": 135, "Low": 128, "Volume": 1000000, ... }
    """
    global model
    if not model:
        return {"error": "Model not loaded"}
    
    # Convert JSON to DataFrame
    df = pd.DataFrame([features])
    
    # Make Prediction
    prediction = model.predict(df)
    return {
        "ticker": TICKER,
        "predicted_price": float(prediction[0]),
        "status": "success"
    }

@app.get("/")
def health_check():
    return {"status": "ML API is running"}
