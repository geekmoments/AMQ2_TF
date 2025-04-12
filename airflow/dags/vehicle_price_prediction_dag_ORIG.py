from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

# Ruta absoluta del archivo DAG como referencia
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 4),
    'retries': 1
}

def run_script(script_name):
    script_path = os.path.join(BASE_PATH, script_name)
    exec(open(script_path).read(), globals())

with DAG('vehicle_price_prediction_dag',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:
    
    echo_task = BashOperator(
        task_id='echo_alive',
        bash_command='echo "Estoy vivo"'
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=run_script,
        op_args=['preprocessing.py']
    )

    training_task = PythonOperator(
        task_id='train_model',
        python_callable=run_script,
        op_args=['training.py']
    )

    validation_task = PythonOperator(
        task_id='validate_model',
        python_callable=run_script,
        op_args=['validation.py']
    )

    echo_task >> preprocess_task >> training_task >> validation_task
