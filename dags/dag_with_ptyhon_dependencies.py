from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'Julieta Basanta',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)}

def get_sklearn():
    import sklearn
    print(f"scikit-learn with version: {sklearn.__version__}")

with DAG(
    default_args=default_args,
    dag_id='dag_with_python_dependencies_v01',
    start_date=datetime(2023,1,1),
    schedule= "@daily"
    ) as dag:

    get_sklearn = PythonOperator(
        task_id='get_sklearn',
        python_callable=get_sklearn)

    get_sklearn
