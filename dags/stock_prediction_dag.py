from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import io
import xgboost as xgb
import matplotlib.pyplot as plt
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---  
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1470476253252288654/GPgUlXbjo_9KPiJPBRD537gAAv4qBwnBWnYHWA8zCQBt__rJ4qrWk_USawt2YSaYkYFI"  # <--- PASTE HERE
TICKER = "NVDA"

with DAG(
    dag_id='stock_prediction_plus_visuals',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'xgboost', 'viz']
) as dag:
    
    @task
    def fetch_stock_data():
        print(f"Fetching data for {TICKER}...")
        df = yf.download(TICKER, period="2y", interval="1d")
        
        # Buffer & Save
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        
        s3 = S3Hook(aws_conn_id='minio_s3_conn')
        filename = f"raw_data/{TICKER}_{datetime.now().date()}.csv"
        s3.load_string(
            string_data=csv_buffer.getvalue(),
            key=filename,
            bucket_name="stock-data",
            replace=True
        )
        return filename
    @task
    def validate_data(file_key):
        s3 = S3Hook(aws_conn_id='minio_s3_conn')
        print(f"Validating data in {file_key}...")
        
        # Read the data
        data = s3.read_key(key=file_key, bucket_name="stock-data")
        df = pd.read_csv(io.StringIO(data))
        
        # Handle the Yahoo Finance header quirk inside validation too
        if "Ticker" in df.columns or "Price" in df.iloc[0].values:
             df = pd.read_csv(io.StringIO(data), header=2)
        
        # RULE 1: Check for Empty Data
        if df.empty:
            raise ValueError("CRITICAL: The downloaded dataset is empty!")
            
        # RULE 2: Check for Missing Values in Critical Columns
        critical_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        for col in critical_cols:
            if col not in df.columns:
                 raise ValueError(f"CRITICAL: Missing column {col}")
            
            # Count missing values
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"WARNING: Found {missing_count} missing values in {col}. Dropping them...")
                # We allow a few missing rows, but not too many
                if missing_count > len(df) * 0.1: # If > 10% are missing, fail.
                    raise ValueError(f"CRITICAL: Too many missing values in {col} (>10%)")
        
        # RULE 3: Statistical Sanity Check (Industrial Math)
        # Price cannot be negative or zero
        if (pd.to_numeric(df['Close'], errors='coerce') <= 0).any():
             raise ValueError("CRITICAL: Found negative or zero prices! Data is corrupted.")
             
        print("✅ Data Quality Check Passed!")
        return file_key

    @task
    def train_and_visualize(file_key):
        s3 = S3Hook(aws_conn_id='minio_s3_conn')
        
        # 1. READ DATA
        data = s3.read_key(key=file_key, bucket_name="stock-data")
        df = pd.read_csv(io.StringIO(data))
        
        # Fix Headers (yfinance quirk)
        if "Ticker" in df.columns or "Price" in df.iloc[0].values:
             df = pd.read_csv(io.StringIO(data), header=2)

        # Force Numeric
        cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Close'])

        # 2. ADVANCED MATH FEATURES (Manual Implementation)
        # SMA
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands (20-day, 2 std dev)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        
        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Target (Tomorrow's Price)
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        # 3. TRAIN MODEL
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI']
        X = df[features]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse} | RMSE: {rmse}")

        # 4. GENERATE PLOT (The Visual)
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual Price', color='blue')
        plt.plot(predictions, label='Predicted Price', color='red', linestyle='--')
        plt.title(f"{TICKER} Price Prediction (RMSE: ${rmse:.2f})")
        plt.legend()
        plt.grid(True)
        
        # Save Plot to Buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Upload Plot to MinIO
        plot_key = f"plots/{TICKER}_prediction_{datetime.now().date()}.png"
        s3.load_bytes(
            bytes_data=img_buffer.getvalue(),
            key=plot_key,
            bucket_name="stock-data",
            replace=True
        )
        print(f"Plot saved to MinIO: {plot_key}")

        # 5. SEND DISCORD ALERT
        if DISCORD_WEBHOOK_URL.startswith("http"):
            payload = {
                "username": "Airflow Bot",
                "content": f"🚀 **Model Trained Successfully!**\n**Ticker:** {TICKER}\n**RMSE Error:** ${rmse:.2f}\n**MSE:** {mse:.2f}\n**Features:** {len(features)}\n*Graph saved to MinIO Data Lake.*"
            }
            requests.post(DISCORD_WEBHOOK_URL, json=payload)

    # DAG Flow
    raw_file = fetch_stock_data()
    validated_file = validate_data(raw_file) 
    train_and_visualize(validated_file)
