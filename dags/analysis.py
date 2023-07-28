from airflow.models import DAG
from airflow import Dataset
from airflow.decorators import task, dag

import pendulum

import pickle

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import precision_score, recall_score, f1_score, plot_roc_curve, accuracy_score

from datetime import timedelta

from includes.vs_modules import analysis_functions, variables

import os
import shutil

import warnings

warnings.filterwarnings('ignore')


# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")

default_args = {
    'owner': 'Julieta Basanta',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)}

my_data = Dataset('/opt/airflow/data/dataset.csv')
my_data_prepared = Dataset('/opt/airflow/data/dataset_prepared.csv')
my_data_cleaned = Dataset('/opt/airflow/data/dataset_cleaned.csv')
my_data_completed = Dataset('/opt/airflow/data/dataset_complete.csv')
my_data_selected= Dataset('/opt/airflow/data/dataset_feature_selected.csv')
my_model = Dataset('/opt/airflow/data/model.pkl')
importances = Dataset('/opt/airflow/data/importances.csv')

@dag(
    default_args=default_args,
    schedule=[my_data], # runs only when dataset is updated               # None -- mejor
    start_date=pendulum.yesterday(),
    catchup=False
    )   
def analysis():
    @task(outlets=[my_data_prepared])
    def prepare_data():
        df = pd.read_csv(my_data_cleaned.uri, 
                    delimiter=',')
        
        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset csv readed')
        task_logger.info(df.head())
        
        # binary columns that should be object type:
        binary_columns = [
            'rv',
            'truck',
            'forgntvl'
        ]
        # change dtype for binary columns to object
        df[binary_columns] = df[binary_columns].astype('object')

        num_cols = df.select_dtypes(include='number').columns

        cat_cols = df.select_dtypes(include='object').columns

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

        # Select only columns without 'Unnamed' inside the column name
        df = df[[col for col in df.columns if 'Unnamed' not in col]]
        
        df.to_csv('/opt/airflow/data/dataset_prepared.csv', index=True)

        task_logger.info('Dataset prepared and updated')
        task_logger.info('Shape:\n')
        task_logger.info(df.shape)

    @task
    def missing_data_analysis():
        df = pd.read_csv(my_data.uri, 
                delimiter=';',
                decimal=',')
        
        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset read')

        # some corrections
        df = df.replace('UNKW', np.nan) # hnd_webcap
        df = df.replace('nan',np.nan) 
        
        task_logger.info('nan replaces done')
        
        # Calculate number of null per variable
        na_per_variable = df.isnull().sum()

        task_logger.info('Calculated number of null per variable')

        # Calculate % of null per variable
        pct_na_per_variable = (na_per_variable / len(df))*100

        task_logger.info('Calculated % of null per variable')

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
        plt.savefig(f"/opt/airflow/data/plots/incomplete_variables-churn_relation.png")

        task_logger.info('Saved incomplete variables churn relation plots')

        # Crear un gr�fico de barras para cada variable
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))

        for i, var in enumerate(variables):
            ax = axs[i//5, i%5]
            sns.countplot(x=var, data=df, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Count')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"/opt/airflow/data/plots/incomplete_variables-distribution.png")

        task_logger.info('Saved incomplete variables distribution plots')


    @task(outlets=[my_data_cleaned])
    def exploratory_data_analysis():
        df = pd.read_csv(my_data.uri, 
                delimiter=';',
                decimal=',')

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        task_logger.info('Dataset read')
        # some corrections
        df = df.replace(['UNKW','nan'], np.nan) 

        task_logger.info('nan replaces done')

        # check if there are columns with varianze equal to zero

        # binary columns that should be object type:
        binary_columns = [
            'rv',
            'truck',
            'forgntvl'
        ]
        # change dtype for binary columns to object
        df[binary_columns] = df[binary_columns].astype('object')

        num_cols_l = df.select_dtypes(include='number').columns
        cat_cols_l = df.select_dtypes(include='object').columns

        num_const = VarianceThreshold(threshold=0)
        num_const.fit(df[num_cols_l])
        num_const_cols = list(set(df[num_cols_l].columns) - set(num_cols_l[num_const.get_support()]))
        cat_const_cols = df[cat_cols_l].nunique()[lambda x: x<2].index.tolist()
        all_const_cols = num_const_cols + cat_const_cols

        task_logger.info('Check if there are columns with varianze = zero')

        # update columns to discard:
        if len(all_const_cols)>0:
            variables.colums_to_drop.append(all_const_cols)
            task_logger.info(f'Columns to discard from the analysis : {all_const_cols}')

        task_logger.info('Updated list of columns to discard')

        
        # BLOCKS OF ANALYSIS: revenue, minutes, calls and other variables
        var_groups = [
            'rev', # revenue
            'mou', # minutes
            'qty', # calls
            'others'
            ]
        task_logger.info('defined blocks for variables analysis')
        
        for term in var_groups:
            analysis_functions.eda(df, term)
            task_logger.info(f'EDA done for {term}')

        for term in ['rev','mou']:
            df = analysis_functions.drop_outliers(df, term)
            task_logger.info(f'Outliers dropped for {term}')

        # Exclude variables -- manually checked
        df = df.drop(variables.variables_to_discard,
                     axis=1)
        task_logger.info('Variables excluded')

        df.to_csv('/opt/airflow/data/dataset_cleaned.csv', index=True)
        task_logger.info('dataset_cleaned saved as csv into airflow/data')

        task_logger.info('Dataset cleaned')
        task_logger.info('Shape:\n')
        task_logger.info(df.shape)

    @task(outlets=[my_data_completed])
    def missing_data_attribution():
        df = pd.read_csv(my_data_prepared.uri, 
                delimiter=',')

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

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

            task_logger.info('Valores de ',col, 'estimados')

        df_without_na = df.dropna()

        if len(df_without_na)!= len(df):
            not_predicted_rows = len(df) - len(df_without_na)
            not_predicted_rows_pct = round(not_predicted_rows/len(df) * 100,2)
            task_logger.warning(f'There is a  {not_predicted_rows_pct} % of rows without predicted values.  Because the amount of missing information was too much to predict missing values.')

        # Select only columns without 'na' inside the column name
        df_without_na = df_without_na[[col for col in df_without_na.columns if 'nan' not in col]]

        df_without_na.to_csv('/opt/airflow/data/dataset_complete.csv', index=True)

        task_logger.info('Dataset completed')
        task_logger.info('Shape:\n')
        task_logger.info(df_without_na.shape)

    @task(outlets=[my_data_selected])
    def feature_selection():
        df = pd.read_csv(my_data_completed.uri, 
                delimiter=',')

        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        X =df.drop('churn', axis=1)
        y = df['churn']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # we use selectbest to get the top features according to MI Classification (MIC).
        # We then employ get_support() to obtain a boolean vector (or mask) which tells us which feature are in the top features and we subset the list features with this mask:
        mic_selection = SelectKBest(mutual_info_classif, k = 'all').fit(X_train,y_train)

        # Get the names of the selected features
        mic_cols = X_train.columns[mic_selection.get_support()].tolist()

        # Get the mutual information scores for the selected features
        mic_scores = mic_selection.scores_[mic_selection.get_support()]

        mic = pd.DataFrame(columns=['cols',
                            'mic_scores'])
        mic['cols'] = mic_cols
        mic['mic_scores'] = mic_scores
        mic = mic.sort_values(by='mic_scores',ascending=False)

        selected_features = mic[mic['mic_scores']>0]

        selected_features_series = pd.Series(selected_features['cols'])
        selected_features_series = pd.concat([selected_features_series, pd.Series('churn')])

        df = df[selected_features_series]
        df.to_csv('/opt/airflow/data/dataset_feature_selected.csv', index=True)

        task_logger.info('Dataset with selected features saved')
        task_logger.info('Shape:\n')
        task_logger.info(df.shape)

    @task(outlets=[my_model])
    def model_training():
        df = pd.read_csv(my_data_selected.uri, 
                delimiter=',')
        
        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        X = df.drop('churn', axis=1)
        y = df['churn']
        

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create model
        model = RandomForestClassifier(random_state=42)
        task_logger.info('created a random forest classifier model')

        # Train the model on the training data
        model.fit(X_train, y_train)
        task_logger.info('trained model')

        # Save the trained model as a pickle file
        with open('/opt/airflow/data/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        task_logger.info('saved model as pickle in output folder')


        # Make predictions on the testing data
        y_pred = model.predict(X_test)
        task_logger.info('predictions made on testing data')

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        task_logger.info('Accuracy:', accuracy)

        # Calculate the precision of the model
        precision = precision_score(y_test, y_pred)
        task_logger.info('Precision:', precision)

        # Calculate the recall of the model
        recall = recall_score(y_test, y_pred)
        task_logger.info('recall:', recall)

        # Calculate the f1 of the model
        f1 = f1_score(y_test, y_pred)
        task_logger.info('f1:', recall)

        # Plot and save the ROC curve
        roc_plot = plot_roc_curve(model, X_test, y_test)
        plt.savefig('/opt/airflow/data/plots/roc_curve.png')

    @task(outlets=[importances])
    def feature_importances():
        model = pickle.load(open(my_model.uri, 'rb'))

        df = pd.read_csv(my_data_selected.uri, 
        delimiter=',')
        
        df.index = df['Customer_ID']
        df = df.drop('Customer_ID', axis=1)

        X = df.drop('churn', axis=1)
        y = df['churn']

        # Get the feature importances from the model
        importances = model.feature_importances_

        # Print and save the feature importances
        importances_df = pd.DataFrame(columns = ['Feature', 'Importance'])
        for i, importance in enumerate(importances):
            row = {'Feature': X.columns[i], 'Importance': importance}
            importances_df = importances_df.append(row, ignore_index=True)

        importances_df = importances_df.sort_values(by='Importance',
                                                    ascending=False)

        importances_df.to_csv('/opt/airflow/data/importances.csv',index=False)

        task_logger.info('Most important features: \n ', importances_df[0:20])

        #################################################################
        # Understand importances with ICE plots
        #################################################################

        # get the indices of the top 10 most important features
        top_indices = importances.argsort()[::-1][:10]
        top_features = X.iloc[:, top_indices]

        # Generate ICE plots for each feature
        features = X.columns.tolist()
        for feature_name in top_features:
            fig, ax = plt.subplots()
            plot_partial_dependence(model, X, [feature_name], ax=ax)
            plt.savefig(f"/opt/airflow/data/plots/{feature_name}_ice_plot.png")
            plt.close()
        
    missing_data_analysis() >> exploratory_data_analysis() >> prepare_data() >> missing_data_attribution() >> feature_selection() >> model_training() >> feature_importances()


analysis()