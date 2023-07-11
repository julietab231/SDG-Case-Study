from airflow.models import DAG
from airflow import Dataset
from airflow.decorators import task
from airflow.operators.python import ExternalPythonOperator
import pendulum

import logging
import pandas as pd
import numpy as np

import os

# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")


my_data = Dataset('/opt/airflow/dags/dataset.csv')

requirements =  Dataset('/opt/airflow/dags/vs_modules/requirements.txt')

with DAG(
    dag_id='prepare_dataset',
    schedule=[my_data], # runs only when dataset is updated
    start_date=pendulum.yesterday(),
    catchup=False):
    
    @task.virtualenv(
        task_id='virtualenv_python',
        requirements=requirements.uri,
        system_site_packages=False
        )
    def callable_virtualenv():
        import sklearn

    @task(outlets=[my_data])
    def prepare_data():
        df = pd.read_csv(my_data.uri, 
                    delimiter=';')

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset csv readed')
        task_logger.info(df.head())

        from sklearn.preprocessing import OneHotEncoder

        num_cols = df.select_dtypes([np.number]).columns
        cat_cols = df.select_dtypes([object, bool]).columns

        # Booleans columns must be categorical
        df[cat_cols] = df[cat_cols].astype(str)

        # Perform one-hot encoding on the selected columns
        onehot_encoder = OneHotEncoder()
        onehot_encoded = onehot_encoder.fit_transform(df[cat_cols])

        # Convert the encoded data into a pandas dataframe and concatenate it with the original dataframe
        onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), 
                                            columns=onehot_encoder.get_feature_names(cat_cols),
                                            index= df.index)
        df = df.merge(onehot_encoded_df, right_index=True, left_index=True)

        # Drop the original categorical columns that were encoded
        df = df.drop(cat_cols, axis=1)

        # change the data type of the selected columns
        df[num_cols] = df[num_cols].astype(float)
        
        df.to_csv('/opt/airflow/dags/dataset_prepared.csv', index=True)

        task_logger.info('Dataset prepared and updated')

    callable_virtualenv() >> prepare_data()