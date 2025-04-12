from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn
import boto3

DATA_PATH = os.path.join("data")
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o limitá a ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del cliente S3 para MinIO
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-east-1'
)

MINIO_BUCKET = 'data'

try:
    file = "XGBoost.pkl"
    s3.download_file(MINIO_BUCKET, file, file)
    with open(file, 'rb') as f:
        model = joblib.load(f)
        
    file = "encoder.pkl"
    s3.download_file(MINIO_BUCKET, file, file)
    with open(file, 'rb') as f:
        encoder = joblib.load(f)

    file = "scaler.pkl"
    s3.download_file(MINIO_BUCKET, file, file)
    with open(file, 'rb') as f:
        scaler = joblib.load(f)

except FileNotFoundError:
    print(f"Error: El archivo '{file}' no fue encontrado.")

# Modelo de datos
class CarFeatures(BaseModel):
    year: int
    km_driven: int
    engine_cc: float
    max_power_bhp: float
    mileage_kmpl: float
    seats: int
    torque_nm: float
    torque_rpm: int
    owner_rank: int
    fuel: str
    seller_type: str
    transmission: str
    brand: str

@app.post("/predict")
def predict_price(features: CarFeatures):
    # Datos categóricos
    cat_data = [[
        features.fuel,
        features.seller_type,
        features.transmission,
        features.brand
    ]]
    cat_encoded = encoder.transform(cat_data)

    # Datos numéricos
    numeric_data = [[
        features.year,
        features.km_driven,
        features.engine_cc,
        features.max_power_bhp,
        features.mileage_kmpl,
        features.seats,
        features.torque_nm,
        features.torque_rpm,
        features.owner_rank
    ]]

    # Escalar solo los numéricos
    numeric_scaled = scaler.transform(numeric_data)

    # Concatenar todo
    full_input = np.hstack((numeric_scaled, cat_encoded))

    # Predecir
    prediction = model.predict(full_input)[0]

    # Convertir a float para la respuesta JSON
    return {"predicted_price": float(prediction)}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
