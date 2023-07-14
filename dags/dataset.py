from airflow.models import DAG
from airflow import Dataset
from airflow.decorators import task
from airflow.operators.python import ExternalPythonOperator
from airflow.operators.bash_operator import BashOperator

import pendulum

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest

from datetime import timedelta

from includes.vs_modules import analysis_functions

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

        df = df.drop(columns=variables_to_drop) # eliminate selected variables with missing values > 30% 

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset csv readed')
        task_logger.info(df.head())

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
        df = pd.read_csv(my_data.uri, 
                delimiter=';',
                decimal=',')

        # some corrections
        df = df.replace('UNKW', np.nan) # hnd_webcap
        df = df.replace('nan',np.nan) 

        # Calculate number of null per variable
        na_per_variable = df.isnull().sum()

        # Calculate % of null per variable
        pct_na_per_variable = (na_per_variable / len(df))*100

        # Show variables with null
        variables_with_na = pct_na_per_variable[pct_na_per_variable > 0].sort_values(ascending=False)

        task_logger.info(variables_with_na)

        task_logger.info('Variables with NULLs - Relation with CHURN variable')
    

        # variables with missing values that we want to compare with 'churn'
        variables = ['numbcars',
                      'dwllsize', # categorical
                      'HHstatin', # categorical
                      'ownrent', # categorical
                      'dwlltype', # categorical
                      'lor',
                      'income',
                      'adults',
                      'infobase', # categorical
                      'hnd_webcap' # categorical
                     ]

        # Create a bar plot for each variable with churn
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))

        for i, var in enumerate(variables):
            churn_by_var = df[['churn', var]].groupby(var).mean()
            ax = axs[i//5, i%5]
            churn_by_var.plot(kind='bar', ax=ax, legend=False)
            ax.set_xlabel(var)
            ax.set_ylabel('churn rate')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"SDG-Case-Study//plots//incomplete_variables-churn_relation.png")

        # Crear un gr�fico de barras para cada variable
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))

        for i, var in enumerate(variables):
            ax = axs[i//5, i%5]
            sns.countplot(x=var, data=df, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"SDG-Case-Study//plots//incomplete_variables-distribution.png")

    @task
    def exploratory_data_analysis():
        df = pd.read_csv(my_data.uri, 
                delimiter=';',
                decimal=',')

        # some corrections
        df = df.replace('UNKW', np.nan) 
        df = df.replace('nan',np.nan) 

        # list of columns to discard: 
        colums_to_drop = []

        # check if there are columns with varianze equal to zero
        num_cols_l = df.select_dtypes(include='number').columns
        cat_cols_l = df.select_dtypes(include='object').columns

        num_const = VarianceThreshold(threshold=0)
        num_const.fit(df[num_cols_l])
        num_const_cols = list(set(df[num_cols_l].columns) - set(num_cols_l[num_const.get_support()]))
        cat_const_cols = df[cat_cols_l].nunique()[lambda x: x<2].index.tolist()
        all_const_cols = num_const_cols + cat_const_cols

        # update columns to discard:
        if len(all_const_cols)>0:
            colums_to_drop.append(all_const_cols)
            task_logger.info(f'Columns to discard from the analysis : {all_const_cols}')
        
        # BLOCKS OF ANALYSIS: revenue, minutes, calls and other variables
        var_groups = ['rev', # revenue
                      'mou', # minutes
                      'qty' # calls
                      ]
        for term in var_groups:
            analysis_functions.eda(df, term)

        for term in ['rev','mou']:
            df = analysis_functions.drop_outliers(df, term)






    @task
    def missing_data_attribution():
        df = pd.read_csv(my_data_prepared.uri, 
                delimiter=',')

        # Calculate number of null per variable
        na_per_variable = df.isnull().sum()

        # Calculate % of null per variable
        pct_na_per_variable = (na_per_variable / len(df))*100

        # Show variables with null
        variables_with_na = pct_na_per_variable[pct_na_per_variable > 0].sort_values(ascending=False)

        # Create a Decision Tree Regressor model 
        model = DecisionTreeRegressor()

        variable_with_na_names = list(variables_with_na.index)

        # For each column with missing values:
        for col in variable_with_na_names:
            # Separate rows with missing values and without for the selected column
            df_with_na =  df.dropna(subset=df.columns.difference([col])) # with na for the selected column
            df_without_na = df.dropna() # without na

            # Train the model for the selected column
            model.fit(df_without_na.drop([col], axis=1), df_without_na[col])

            # Predict missing values for the selected column
            df.loc[df_with_na.index,col] = model.predict(df_with_na.drop([col], axis=1))

            print('Valores de ',col, 'estimados')

        df_without_na = df.dropna()

        if len(df_without_na)!= len(df):
            not_predicted_rows = len(df) - len(df_without_na)
            not_predicted_rows_pct = round(not_predicted_rows/len(df) * 100,2)
            #task_logger.warning(f'There is a  {not_predicted_rows_pct} % of rows without predicted values.  Because the amount of missing information was too much to predict missing values.')

        df_without_na.to_csv('/opt/airflow/dags/dataset_complete.csv', index=True)

missing_data_analysis() >> prepare_data() >> missing_data_attribution