from airflow.models import DAG
from airflow import Dataset
from airflow.decorators import task
from airflow.operators.python import PythonOperator
import pendulum
from airflow.models.baseoperator import chain

import sys
sys.path.insert(1,'/SDG-Case-Study/dags')

import logging
import pandas as pd

# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")

my_data = Dataset('/opt/airflow/dags/dataset_prepared.csv')

with DAG(
    dag_id='exploratory-data-analysis',
    schedule=[my_data],
    start_date=pendulum.yesterday(),
    catchup=False
):
    @task
    def exploratory_data_analysis():
        df = pd.read_csv(my_data.uri, 
                    delimiter=';')
        task_logger.info(df.head)
        task_logger.info('exploratory data analysis')
    exploratory_data_analysis() 


