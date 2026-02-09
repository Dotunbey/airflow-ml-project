from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
import yfinance as yf
import io
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define the Pipeline
with DAG(
    dag_id='end_to_end_stock_prediction',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'xgboost']
) as dag:

    # TASK 1: Ingest Data (ETL)
    @task
    def fetch_stock_data():
        # 1. Fetch from Yahoo Finance
        ticker = "AAPL"
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, period="2y", interval="1d")
        
        # 2. Prepare for storage (Buffer)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        
        # 3. Save to MinIO (Your Data Lake)
        filename = f"raw_data/{ticker}_{datetime.now().date()}.csv"
        s3 = S3Hook(aws_conn_id='minio_s3_conn') # Uses the connection we created
        s3.load_string(
            string_data=csv_buffer.getvalue(),
            key=filename,
            bucket_name="stock-data",
            replace=True
        )
        print(f"Saved to s3://stock-data/{filename}")
        return filename

    # TASK 2: Train Model (ML)
    @task
    def train_xgboost_model(file_key):
        s3 = S3Hook(aws_conn_id='minio_s3_conn')
        
        # 1. Read Data from Data Lake
        print(f"Reading {file_key} from MinIO...")
        data = s3.read_key(key=file_key, bucket_name="stock-data")
        
        # --- FIX STARTS HERE ---
        # Skip the first two rows if they contain ticker info (common in yfinance)
        # We try reading normally, then force numeric conversion
        df = pd.read_csv(io.StringIO(data))
        
        # If yfinance returned a multi-level header (e.g., Price | Ticker), fix it:
        if "Ticker" in df.columns or "Price" in df.iloc[0].values:
             df = pd.read_csv(io.StringIO(data), header=2) # Skip bad headers

        # Ensure 'Close' is numeric. Coerce errors (like 'AAPL') to NaN
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Drop any rows that failed conversion
        df = df.dropna(subset=['Close'])
        # --- FIX ENDS HERE ---
        
        # 2. Feature Engineering (The Math)
        # Create a 'Target': Predict tomorrow's Close price
        df['Target'] = df['Close'].shift(-1)
        # Create Features: Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df = df.dropna()
        
        # 3. Train XGBoost
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50']
        # Ensure all feature columns are numeric too
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna() # Final clean
        
        X = df[features]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        model.fit(X_train, y_train)
        
        # 4. Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model Training Complete. MSE: {mse}")
        
        # 5. Save the 'Model' back to S3
        metrics = f"Model Trained on {datetime.now()}\nMSE: {mse}\nFeatures: {features}"
        s3.load_string(
            string_data=metrics,
            key=f"models/xgboost_metrics_{datetime.now().date()}.txt",
            bucket_name="stock-data",
            replace=True
        )

    # Define Flow
    data_file = fetch_stock_data()
    train_xgboost_model(data_file)
