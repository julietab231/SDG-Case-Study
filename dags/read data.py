
import json
import subprocess

import pendulum

import pandas as pd

from airflow.decorators import dag, task
from airflow import Dataset
import logging

# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")

my_data = Dataset('/opt/airflow/dags/dataset.csv')

@dag(
    schedule=[my_data],
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["analysis"],
)
def read_data():
    """
    ### TaskFlow API Analysis Documentation
    This is a first descriptive analysis of data.
    """
    @task()
    def extract():
        """
        #### read data

        """
        # Define the name of the CSV file
        file_name = "SDG-Case-Study//dataset.csv"

        # Read the CSV file from the mounted volume using Pandas
        df = pd.read_csv(file_name,
                         delimiter=';',
                         error_bad_lines=False)
        task_logger.info('Dataset csv readed')
        task_logger.info(df.head())

        return df

    @task()
    def import_db():
        """
        import dataset in sqlite db
        """
        cmd = '''sqlite3 dataset.db 
                -cmd ".output /db/dataset/mysession.txt"
                -cmd ".log /db/dataset/mycommands.sql"
                -cmd ".read /db/dataset/schema.sql" 
                -cmd ".mode csv" 
                -cmd ".separator ;" 
                -cmd ".import dataset.csv dataset" 
                -cmd ".exit" '''
        output = subprocess.check_output(cmd, shell=True)

        task_logger.info('Dataset imported into dataset.db sqlite3 ')
        task_logger.info(output.decode('utf-8'))
        return output.decode('utf-8')

    @task(multiple_outputs=True)

    def summarise(df):
        """
        #### Descriptive analysis
        """

        # Print head of data
        print('Head:\n',df.head())

        # check the shape of the DataFrame
        print('\nShape of the DataFrame:\n', df.shape)  

        # check the data types of each column
        df_dtypes = df.dtypes
        df_dtypes.to_csv('SDG-Case-Study//df_dtypes.csv')
        print('\nData types of each column:\n', df.dtypes)

        # get summary statistics of the numerical/date format column
        df_describe = df.describe()
        df_describe.to_csv('SDG-Case-Study//df_describe.csv')
        print('\nSummary statistics \n',df.describe()) 

        # count the number of unique values in each column
        print('\nUnique values in each column:\n', df.nunique())

        # check the frequency distribution of the 'churn' column
        print('\nDistribution of the churn column:\n', df['churn'].value_counts())

        return {"Descriptive statistics": df.describe()}

    df = extract()
    summarise(df)

read_data()


