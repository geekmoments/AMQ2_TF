import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import mean_squared_error

import boto3
import joblib
import io 
from io import BytesIO
import pickle

DATA_PATH = os.path.join("data")
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

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
    filename=os.path.join(LOG_PATH, 'validation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def validation_main():
    models = ['Ridge','RandomForest','XGBoost','LightGBM']
    print("Comienza la valdiacion...")
    #best_score = -np.inf
    name_model = None
    best_name = ""

    for name in models:
        def download_npy_from_minio(key_name):
            with io.BytesIO() as f:
                s3.download_fileobj(MINIO_BUCKET, key_name, f)
                f.seek(0)
                array = np.load(f)
            return array

        # Usá las funciones
        X_test = download_npy_from_minio('X_test.npy')
        y_test = download_npy_from_minio('y_test.npy')

        file = name + ".pkl"
        file_path = os.path.join(MODEL_PATH, file)        

        print(f"Descargo modelo {file_path}")

        # Descargar el archivo a un objeto en memoria
        s3.download_file(MINIO_BUCKET, file, file)

        # Cargar el modelo desde el archivo descargado
        try:
            with open(file, 'rb') as f:
                model = joblib.load(f)
        except FileNotFoundError:
            print(f"Error: El archivo '{file}' no fue encontrado.")
        except pickle.UnpicklingError:
            print(f"Error: No se pudo deserializar el archivo '{file}'. ¿Está corrupto o no es un archivo pickle válido?")
        except PermissionError:
            print(f"Error: No tenés permiso para leer el archivo '{file}'.")
        except Exception as e:
            print(f"Error inesperado al cargar el archivo '{file}': {e}")
        
        print(f"Ejecuto modelo {name}")
        y_pred = model.predict(X_test)
        print(f"Evaluo metricas del modelo {name}")
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        logging.info(f"Modelo: {name} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
        print(f"Modelo: {name} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
        logging.info(f"Modelo {name} validado.")

    return


if __name__ == "__main__":
    validation_main()
