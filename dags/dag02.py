################################################
################################################
################################################
# advanced analysis
################################################
################################################
################################################


################################################
# install libraries
################################################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

################################################
# read data
################################################

# Define the name of the CSV file
file_name = "SDG-Case-Study//dataset.csv"

# Read the CSV file from the mounted volume using Pandas
df = pd.read_csv(file_name,
                 delimiter=';',
                 error_bad_lines=False)
print('data imported')
################################################
# data tranformation
################################################

# save the customer_id as index, as it's not needed for analysis
df.index = df['Customer_ID']
df = df.drop('Customer_ID', axis=1)
print('Customer_ID saved as index')

# Split the data into features (X) and target (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('data splited into training and testing sets')

# Create a decision tree classifier model
model = DecisionTreeClassifier(random_state=42)
print('created a decision tree classifier model')

# Train the model on the training data
model.fit(X_train, y_train, verbose=1)
print('trained model')

# save the model to disk
model.save('SDG-Case-Study//tree_classifier_model.h5')
print('trained model saved as "tree_classifier_model.h5"')

# Make predictions on the testing data
y_pred = model.predict(X_test)
print('predictions made on testing data')

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Get the feature importances from the model
importances = model.feature_importances_

# Print the feature importances
for i, importance in enumerate(importances):
    print(f'{X.columns[i]}: {importance}')