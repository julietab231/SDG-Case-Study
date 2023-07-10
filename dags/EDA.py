
import subprocess

cmd = '''pip install -r SDG-Case-Study/requirements.txt'''
output = subprocess.check_output(cmd, shell=True)

import json
import pendulum

import pandas as pd
import numpy as np

from airflow.decorators import dag, task
import logging

from sklearn.preprocessing import OneHotEncoder

# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")


@dag(
    schedule=None,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["analysis","EDA"],
)
def exploratory_data_analysis():
    """
    ### TaskFlow API Exploratory Data Analysis
    """

    @task()
    def read_data():
        """
        #### read data

        """
        # Define the name of the CSV file
        file_name = "SDG-Case-Study//dataset.csv"

        # Read the CSV file from the mounted volume using Pandas
        df = pd.read_csv(file_name,
                         delimiter=';',
                         error_bad_lines=False)

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset csv readed')
        task_logger.info(df.head())

        return df

    @task()
    def prepare():
        df = read_data()

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

        df.to_csv('output//prepared_dataset.csv', index=True)

        return df
    prepare()
exploratory_data_analysis()