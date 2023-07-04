################################################
################################################
################################################
# Use trained model
################################################
################################################
################################################

'''
At first run this comand in your environment:
pip install -r requirements.txt
'''

################################################
# install libraries
################################################
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import random


################################################
# Load the saved model from the pickle file
################################################
with open('SDG-Case-Study//output//dt_model.pkl', 'rb') as f:
    model = pickle.load(f)

################################################
# Read customer data
################################################

# Define the name of the CSV file
file_name = "SDG-Case-Study//dataset.csv"

# Read the CSV file from the mounted volume using Pandas
df = pd.read_csv(file_name,
                 delimiter=';',
                 decimal=',',
                 error_bad_lines=False)

################################################
# Select n customers for the prediction
################################################

n_customers = input("What is the number of customers you would like to analyze?\n")
if n_customers.isdigit():
    if int(n_customers) <0: 
        print("Please enter a number between 1 to 10.000")
    else:
        print("Selecting ",n_customers," customers from dataset \n")
else: 
    print("Please enter a number between 1 to 10.000")

X = pd.read_csv('SDG-Case-Study//dataset_processed_X.csv', index_col=0)
y = pd.read_csv('SDG-Case-Study//dataset_processed_y.csv', index_col=0)


X = X.sample(n=int(n_customers),random_state=42)
y = y.sample(n=int(n_customers),random_state=42)

################################################
# Apply random forest model
################################################
print("Applying random forest model")

# Make predictions on the testing data
churn_predictions  = model.predict(X)

churn_predictions = pd.DataFrame(churn_predictions, 
                                 index= X.index)
print(churn_predictions)
churn_predictions.to_csv('SDG-Case-Study//output//dag03_result_churn.csv',index=True)