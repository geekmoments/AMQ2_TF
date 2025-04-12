import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
#from sklearn.metrics import make_scorer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
#import xgboost as xgb
#import lightgbm as lgb
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from scipy import stats
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer


from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

import joblib
import logging

import boto3
import io


DATA_PATH = os.path.join("data")
MODEL_PATH = os.path.join("models")
LOG_PATH = os.path.join("logs")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


import boto3
import io

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
    filename=os.path.join(LOG_PATH, 'preprocessing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATA_PATH = os.path.join("")
DATA_PATH = '/opt/airflow/dags/'


def load_datasets():
    datasets = {
        'car_data': pd.read_csv(os.path.join(DATA_PATH, 'car_data.csv')),
        'car_details': pd.read_csv(os.path.join(DATA_PATH, 'CAR_DETAILS_FROM_CAR_DEKHO.csv')),
        'car_details_v4': pd.read_csv(os.path.join(DATA_PATH, 'car_details_V4.csv')),
        'car_details_v3': pd.read_csv(os.path.join(DATA_PATH, 'Car_details_V3.csv')),
    }
    return datasets

# LIMPIEZA BÁSICA DEL DATASET
def basic_cleaning(df):
    print("\n2. LIMPIEZA BÁSICA DE DATOS")
    df_clean = df.copy()

    duplicados_iniciales = df_clean.duplicated().sum()
    df_clean.drop_duplicates(inplace=True)
    print(f"\nA. Duplicados eliminados: {duplicados_iniciales}")

    # Separar 'name' en 'brand' y 'model'
    df_clean['brand'] = df_clean['name'].str.split().str[0]  # Primera palabra como brand
    df_clean['model'] = df_clean['name'].str.split(n=1).str[1]  # Resto como model
    print("\nB. Columna 'name' separada en 'brand' y 'model'.")

    df_clean['engine_cc'] = df_clean['engine'].str.extract('(\d+)').astype(float)

    df_clean['max_power_bhp'] = df_clean['max_power'].str.extract('(\d+\.?\d*)').astype(float)
    df_clean['max_power_bhp'] = df_clean['max_power_bhp'].replace(0, np.nan)

    def standardize_mileage(x):
        if pd.isna(x):
            return np.nan
        if 'km/kg' in str(x):
            value = float(str(x).split()[0])
            return value * 1.39
        return float(str(x).split()[0])

    df_clean['mileage_kmpl'] = df_clean['mileage'].apply(standardize_mileage)

    def extract_torque_values(x):
        if pd.isna(x):
            return np.nan, np.nan

        patterns = {
            'nm_standard': r'(\d+\.?\d*)\s*Nm',
            'nm_at': r'(\d+\.?\d*)\s*Nm\s*@',
            'kgm_standard': r'(\d+\.?\d*)\s*kgm',
            'kgm_at': r'(\d+\.?\d*)\s*kgm\s*@',
            'nm_parenthesis': r'\((\d+\.?\d*)\s*Nm\)',
            'kgm_parenthesis': r'\((\d+\.?\d*)\s*kgm\)',
            'bare_number_at': r'^(\d+\.?\d*)\s*@',
            'number_with_at': r'(\d+\.?\d*)\s*@',
            'kgm_at_rpm': r'(\d+\.?\d*)\s*\(kgm@\s*rpm\)',
            'nm_range': r'(\d+\.?\d*)-(\d+\.?\d*)\s*Nm',
            'kgm_range': r'(\d+\.?\d*)-(\d+\.?\d*)\s*kgm',
        }

        rpm_patterns = {
            'standard': r'@\s*(\d+)(?:\s*rpm)?',
            'range': r'@\s*(\d+)-(\d+)(?:\s*rpm)?',
            'parenthesis': r'\((\d+)\s*rpm\)',
            'range_parenthesis': r'\((\d+)-(\d+)\s*rpm\)',
            'at_rpm': r'@\s*(\d+)\s*\(rpm\)',
        }

        torque_value = np.nan
        rpm_value = np.nan
        x_lower = str(x).lower().strip()

        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, x_lower, re.IGNORECASE)
            if match:
                if 'range' in pattern_name:
                    torque_value = float(match.group(2))
                else:
                    torque_value = float(match.group(1))
                if 'kgm' in pattern_name:
                    torque_value *= 9.80665
                break

        for pattern_name, pattern in rpm_patterns.items():
            match = re.search(pattern, x_lower, re.IGNORECASE)
            if match:
                if 'range' in pattern_name:
                    min_rpm = float(match.group(1))
                    max_rpm = float(match.group(2))
                    rpm_value = (min_rpm + max_rpm) / 2
                else:
                    rpm_value = float(match.group(1))
                break

        return torque_value, rpm_value

    # Extracción de torque
    df_clean[['torque_nm', 'torque_rpm']] = pd.DataFrame(
        df_clean['torque'].apply(extract_torque_values).tolist(),
        index=df_clean.index
    )

    print("\nB. Conversiones realizadas: engine_cc, max_power_bhp, mileage_kmpl, torque_nm, torque_rpm")

    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }
    df_clean['owner_rank'] = df_clean['owner'].map(owner_map)
    print("\nC. Owner convertido a ranking numérico")

    # NO se imputan nulos aquí; la imputación debe hacerse posteriormente, después de separar los sets de entrenamiento y prueba
    # Comentado para evitar imputación anticipada
    # numeric_cols = ['engine_cc', 'max_power_bhp', 'mileage_kmpl', 'seats', 'torque_nm', 'torque_rpm']
    # for col in numeric_cols:
    #     df_clean[col].fillna(df_clean[col].median(), inplace=True)

    print("\nD. Valores nulos pendientes de imputación posterior")
    print("Valores nulos restantes:")
    print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

    print("\nE. Ejemplos de procesamiento de torque:")
    print(df_clean[['torque', 'torque_nm', 'torque_rpm']].sample(5, random_state=42))

    return df_clean


def prepare_data(df_clean, target_column='selling_price'):
    """Prepara los datos para el modelado."""

    # Separar features y target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir columnas numéricas y categóricas
    numeric_cols = ['year', 'km_driven', 'engine_cc', 'max_power_bhp',
                   'mileage_kmpl', 'seats', 'torque_nm', 'torque_rpm', 'owner_rank']
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'brand']

    # Imputar valores faltantes
    imputer = IterativeImputer(random_state=42)
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # Codificar variables categóricas
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_train = encoder.fit_transform(X_train[categorical_cols])
    encoded_test = encoder.transform(X_test[categorical_cols])

        # Convertir codificación a DataFrame
    encoded_train = pd.DataFrame(
        encoded_train,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index
    )
    encoded_test = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index
    )

    # Concatenar datos numéricos y categóricos
    X_train = pd.concat([X_train[numeric_cols], encoded_train], axis=1)
    X_test = pd.concat([X_test[numeric_cols], encoded_test], axis=1)

    # Escalar datos numéricos
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Se generan objetos encoder y scaler para poder consultar por API
    try:
        file = "encoder.pkl"
        file_path = os.path.join(MODEL_PATH, file)
        joblib.dump(encoder, file_path)
        s3.upload_file(file_path, MINIO_BUCKET, file)
        logging.info(f"Archivo encoder.pkl creado y subido.")
        file = "scaler.pkl"
        file_path = os.path.join(MODEL_PATH, file)
        joblib.dump(scaler, file_path)
        s3.upload_file(file_path, MINIO_BUCKET, file)
        logging.info(f"Archivo scaler.pkl creado y subido.")
    except Exception as e:
        logging.error(f"Error al crear scaler.pkl {name}: {e}")


    return X_train, X_test, y_train, y_test


def define_metrics():
    """Define las métricas de evaluación."""
    return {
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'R2': r2_score
    }

def evaluate_model(y_true, y_pred, metrics):
    """Evalúa el modelo usando las métricas definidas."""
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}


def preprocess_main():
    datasets = load_datasets()
    df = datasets['car_details_v3']  # SE SELECCIONA SOLO V3 POR SER EL DATASET MAS CONSISTENTE Y MAS GRANDE (Por lo analizado en el TP de AdMI)
    
    #LIMPIEZA DE DATOS
    df_clean = basic_cleaning(df)

    # Configuración de MinIO
    MINIO_ENDPOINT = 'S3:9000'  # o la IP/host donde corre MinIO
    MINIO_ACCESS_KEY = 'minio'
    MINIO_SECRET_KEY = 'minio123'
    MINIO_BUCKET = 'data'

    # Cliente MinIO vía boto3
    s3 = boto3.client(
        's3',
        endpoint_url=f'http://{MINIO_ENDPOINT}',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

    # PREPARACIÓN DE LOS DATOS
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    # Función para guardar en MinIO usando BytesIO
    def upload_npy_to_minio(array, key_name):
        with io.BytesIO() as f:
            np.save(f, array)
            f.seek(0)
            s3.upload_fileobj(f, MINIO_BUCKET, key_name)

    # Guardar los archivos en el bucket "data"
    upload_npy_to_minio(X_train, 'X_train.npy')
    upload_npy_to_minio(y_train, 'y_train.npy')
    upload_npy_to_minio(X_test, 'X_test.npy')
    upload_npy_to_minio(y_test, 'y_test.npy')


    #PRUEBA FIN


    #pd.to_pickle(scaler, os.path.join('models', 'scaler.pkl'))
    #pd.to_pickle(encoder, os.path.join('models', 'encoder.pkl'))
    return 0

if __name__ == "__main__":
   print("\nSE EJECUTA PREPROCESS!!!!")
   preprocess_main()



