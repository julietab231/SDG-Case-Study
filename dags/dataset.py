from airflow.models import DAG
from airflow import Dataset
from airflow.decorators import task
from airflow.operators.python import ExternalPythonOperator
from airflow.operators.bash_operator import BashOperator

import pendulum

import logging
import pandas as pd
import numpy as np

from datetime import timedelta

from includes.vs_modules import variables

import os

# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")

default_args = {
    'owner': 'Julieta Basanta',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)}

my_data = Dataset('/opt/airflow/dags/dataset.csv')
my_data_prepared = Dataset('/opt/airflow/dags/dataset_prepared.csv')

with DAG(
    default_args=default_args,
    dag_id='exploratory_data_analysis',
    schedule=[my_data], # runs only when dataset is updated
    start_date=pendulum.yesterday(),
    catchup=False):

    @task(outlets=[my_data])
    def prepare_data():
        df = pd.read_csv(my_data.uri, 
                    delimiter=';',
                    decimal=',')

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset csv readed')
        task_logger.info(df.head())

        from sklearn.preprocessing import OneHotEncoder

        # num_cols = df.select_dtypes([np.number]).columns
        # cat_cols = df.select_dtypes([object, bool]).columns

        num_cols = variables.num_cols

        cat_cols = variables.cat_cols

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
        task_logger.info('Shape:\n')
        task_logger.info(df.shape)

    @task
    def missing_data_analysis():
        df = pd.read_csv(my_data_prepared.uri, 
                delimiter=',')

        # Calculate number of null per variable
        na_per_variable = df.isnull().sum()

        # Calculate % of null per variable
        pct_na_per_variable = (na_per_variable / len(df))*100

        # Show variables with null
        variables_with_na = pct_na_per_variable[pct_na_per_variable > 0]
        task_logger.info(variables_with_na)
    

    prepare_data() >> missing_data_analysis() 