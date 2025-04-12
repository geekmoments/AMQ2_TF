from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/opt/airflow/data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/opt/airflow/logs')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/opt/airflow/dags')))

# Importación directa de funciones principales desde tus scripts
from preprocessing import preprocess_main as preprocess_main
from training import training_main as training_main
from validation import validation_main as validation_main

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 4),
    'retries': 1,
    'depends_on_past': False,
    'email_on_failure': False
}

with DAG(
    dag_id='vehicle_price_prediction_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    echo_task = BashOperator(
        task_id='echo_alive',
        bash_command='echo "Estoy vivo"'
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_main
    )

    training_task = PythonOperator(
        task_id='train_model',
        python_callable=training_main
    )

    validation_task = PythonOperator(
        task_id='validate_model',
        python_callable=validation_main
    )


    echo_task >> preprocess_task >> training_task >> validation_task
