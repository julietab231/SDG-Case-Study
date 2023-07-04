################################################
################################################
################################################
# Use trained model
################################################
################################################
################################################


################################################
# install libraries
################################################
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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

X = pd.read_csv('SDG-Case-Study//dataset_processed_X.csv')
y = pd.read_csv('SDG-Case-Study//dataset_processed_y.csv')

X.index = df['Customer_ID']
y.index = df['Customer_ID']

selected_customers = random.sample(range(min(X.index), max(X.index)), 
                                   int(n_customers))

X = X.iloc[selected_customers]
y = y.iloc[selected_customers]

################################################
# Apply random forest model
################################################
print("Applying random forest model")

# Make predictions on the testing data
churn_predictions  = model.predict(X)


# Calculate variable importance for the new data
importances = model.feature_importances_
importances_df = pd.DataFrame(columns=['Feature', 'Importance'])
for i, importance in enumerate(importances):
    row = {'Feature': X.columns[i], 'Importance': importance}
    importances_df = importances_df.append(row, ignore_index=True)

importances_df = importances_df.sort_values(by='Importance',
                                            ascending=False)

# Combine the predictions and variable importance into a single dataframe
results_df = pd.DataFrame({
    'Customer ID': X.index,  # Assuming customer ID is data index
    'Churn Prediction': churn_predictions,
    'Variable Importance': importances_df['Importance']
})

# Print the results dataframe
print(results_df)


results_df.to_csv('SDG-Case-Study//output//dag03_result.csv',index=False)