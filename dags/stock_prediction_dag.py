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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1470476253252288654/GPgUlXbjo_9KPiJPBRD537gAAv4qBwnBWnYHWA8zCQBt__rJ4qrWk_USawt2YSaYkYFI"
TICKER = "NVDA"

with DAG(
    dag_id='stock_prediction_plus_visuals',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'xgboost', 'production']
) as dag:

    @task
    def fetch_stock_data():
        print(f"Fetching 5 years of data for {TICKER}...")
        # UPGRADE: Fetching 5 years for better accuracy
        df = yf.download(TICKER, period="5y", interval="1d")
        
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
        
        data = s3.read_key(key=file_key, bucket_name="stock-data")
        df = pd.read_csv(io.StringIO(data))
        
        if "Ticker" in df.columns or "Price" in df.iloc[0].values:
             df = pd.read_csv(io.StringIO(data), header=2)
        
        # RULE 1: Check for Empty Data
        if df.empty:
            raise ValueError("CRITICAL: The downloaded dataset is empty!")
            
        # RULE 2: Check for Missing Values
        critical_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        for col in critical_cols:
            if col not in df.columns:
                 raise ValueError(f"CRITICAL: Missing column {col}")
            if df[col].isnull().sum() > len(df) * 0.1:
                raise ValueError(f"CRITICAL: Too many missing values in {col}")
        
        # RULE 3: Statistical Sanity Check
        if (pd.to_numeric(df['Close'], errors='coerce') <= 0).any():
             raise ValueError("CRITICAL: Found negative or zero prices!")
             
        print("✅ Data Quality Check Passed!")
        return file_key

    @task
    def train_and_visualize(file_key):
        s3 = S3Hook(aws_conn_id='minio_s3_conn')
        
        # 1. READ & PREP DATA
        print(f"Reading {file_key} from MinIO...")
        data = s3.read_key(key=file_key, bucket_name="stock-data")
        df = pd.read_csv(io.StringIO(data))
        
        if "Ticker" in df.columns or "Price" in df.iloc[0].values:
             df = pd.read_csv(io.StringIO(data), header=2)

        cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Close'])

        # 2. FEATURE ENGINEERING
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        # 3. TRAIN WITH GRID SEARCH (The "Brain")
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI']
        X = df[features]
        y = df['Target']
        
        split_point = int(len(df) * 0.85)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        print("Starting Grid Search...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
        }
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_squared_error',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_ # <--- This defines the Best Model
        print(f"✅ Best Params: {grid_search.best_params_}")

        # 4. EVALUATE
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # 5. SAVE PLOT
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual Price', color='blue')
        plt.plot(predictions, label='Predicted Price', color='red', linestyle='--')
        plt.title(f"{TICKER} Prediction (RMSE: ${rmse:.2f})")
        plt.legend()
        plt.grid(True)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        plot_key = f"plots/{TICKER}_prediction_{datetime.now().date()}.png"
        s3.load_bytes(
            bytes_data=img_buffer.getvalue(),
            key=plot_key,
            bucket_name="stock-data",
            replace=True
        )

        # 6. SAVE MODEL ARTIFACT (For the API - Phase 3)
        local_model_path = f"/tmp/{TICKER}_model.json"
        best_model.save_model(local_model_path)
        
        s3.load_file(
            filename=local_model_path,
            key=f"models/production_model_{TICKER}.json", 
            bucket_name="stock-data",
            replace=True
        )
        print(f"✅ Model Artifact saved for API.")

        # 7. SEND DISCORD ALERT
        if DISCORD_WEBHOOK_URL:
            # Predict tomorrow
            latest_data = X.iloc[[-1]]
            tomorrow_pred = best_model.predict(latest_data)[0]
            today_price = df.iloc[-1]['Close']
            pct_change = ((tomorrow_pred - today_price) / today_price) * 100
            direction = "🟢 UP" if pct_change > 0 else "🔴 DOWN"
            
            payload = {
                "username": "Airflow Bot",
                "content": f"""🚀 **Daily Stock Forecast**
**Ticker:** {TICKER}
**Current Price:** ${today_price:.2f}
**Predicted Tomorrow:** ${tomorrow_pred:.2f}
**Potential Move:** {direction} ({pct_change:.2f}%)
**Model Confidence (RMSE):** ${rmse:.2f}
**Best Params:** `{grid_search.best_params_}`"""
            }
            try:
                requests.post(DISCORD_WEBHOOK_URL, json=payload)
            except Exception as e:
                print(f"Discord Error: {e}")

        # 8. EXPERIMENT TRACKING (The Lab Notebook - Phase 2)
        history_file_key = "experiments/model_history.csv"
        new_record = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": TICKER,
            "Model": "XGBoost",
            "RMSE": round(rmse, 2),
            "MSE": round(mse, 2),
            "Best_Params": str(grid_search.best_params_),
            "Features": str(features)
        }
        new_df = pd.DataFrame([new_record])
        
        try:
            obj = s3.read_key(key=history_file_key, bucket_name="stock-data")
            history_df = pd.read_csv(io.StringIO(obj))
            history_df = pd.concat([history_df, new_df], ignore_index=True)
        except Exception:
            history_df = new_df
            
        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)
        s3.load_string(
            string_data=csv_buffer.getvalue(),
            key=history_file_key,
            bucket_name="stock-data",
            replace=True
        )
        print("✅ Experiment Log Updated!")

    # DAG Flow
    raw_file = fetch_stock_data()
    validated_file = validate_data(raw_file)
    train_and_visualize(validated_file)
