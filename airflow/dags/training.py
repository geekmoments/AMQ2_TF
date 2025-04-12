import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import logging


DATA_PATH = os.path.join("data")
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


import boto3
import io
import numpy as np

# Configuración del cliente S3 para MinIO
s3 = boto3.client(
    's3',
    endpoint_url='http://S3:9000',  # O la URL que estés usando
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-east-1'
)

MINIO_BUCKET = 'data'

logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():

    def download_npy_from_minio(key_name):
        with io.BytesIO() as f:
            s3.download_fileobj(MINIO_BUCKET, key_name, f)
            f.seek(0)
            array = np.load(f)
        return array

    # Usá las funciones
    X_train = download_npy_from_minio('X_train.npy')
    y_train = download_npy_from_minio('y_train.npy')
    return (X_train, y_train)


def train_models(X_train, y_train):
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }

    name_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)

        file = name + ".pkl"
        file_path = os.path.join(MODEL_PATH, file)

        try:
            joblib.dump(model, file_path)
            s3.upload_file(file_path, MINIO_BUCKET, file)
            logging.info(f"Modelo {name} creado y subido.")
        except Exception as e:
            logging.error(f"Error al procesar el modelo {name}: {e}")

    return

def training_main():
    X_train, y_train = load_data()
    train_models(X_train, y_train)
    return

if __name__ == "__main__":
    training_main()
    