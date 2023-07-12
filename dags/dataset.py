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

import os

# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")

default_args = {
    'owner': 'Julieta Basanta',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)}

my_data = Dataset('/opt/airflow/dags/dataset.csv')

with DAG(
    default_args=default_args,
    dag_id='prepare_dataset',
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

        num_cols = [
            'rev_Mean',
            'mou_Mean',
            'totmrc_Mean',
            'da_Mean',
            'ovrmou_Mean',
            'ovrrev_Mean',
            'vceovr_Mean',
            'datovr_Mean',
            'roam_Mean',
            'change_mou',
            'change_rev',
            'drop_vce_Mean',
            'drop_dat_Mean',
            'blck_vce_Mean',
            'blck_dat_Mean',
            'unan_vce_Mean',
            'unan_dat_Mean',
            'plcd_vce_Mean',
            'plcd_dat_Mean',
            'recv_vce_Mean',
            'recv_sms_Mean',
            'comp_vce_Mean',
            'comp_dat_Mean',
            'custcare_Mean',
            'ccrndmou_Mean',
            'cc_mou_Mean',
            'inonemin_Mean',
            'threeway_Mean',
            'mou_cvce_Mean',
            'mou_cdat_Mean',
            'mou_rvce_Mean',
            'owylis_vce_Mean',
            'mouowylisv_Mean',
            'iwylis_vce_Mean',
            'mouiwylisv_Mean',
            'peak_vce_Mean',
            'peak_dat_Mean',
            'mou_peav_Mean',
            'mou_pead_Mean',
            'opk_vce_Mean',
            'opk_dat_Mean',
            'mou_opkv_Mean',
            'mou_opkd_Mean',
            'drop_blk_Mean',
            'attempt_Mean',
            'complete_Mean',
            'callfwdv_Mean',
            'callwait_Mean',
            'months',
            'uniqsubs',
            'actvsubs',
            'totcalls',
            'totmou',
            'totrev',
            'adjrev',
            'adjmou',
            'adjqty',
            'avgrev',
            'avgmou',
            'avgqty',
            'avg3mou',
            'avg3qty',
            'avg3rev',
            'avg6mou',
            'avg6qty',
            'avg6rev',
            'hnd_price',
            'phones',
            'models',
            'lor',
            'adults',
            'income',
            'numbcars',
            'forgntvl',
            'eqpdays',
            'churn'
    ]
        cat_cols = [
            'new_cell',
            'crclscod',
            'asl_flag',
            'prizm_social_one',
            'area',
            'dualband',
            'refurb_new',
            'hnd_webcap',
            'ownrent',
            'dwlltype',
            'marital',
            'infobase',
            'HHstatin',
            'dwllsize',
            'ethnic',
            'kid0_2',
            'kid3_5',
            'kid6_10',
            'kid11_15',
            'kid16_17',
            'creditcd',
            'rv',
            'truck'
            ]

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
        task_logger.info(df.shape())

    prepare_data()