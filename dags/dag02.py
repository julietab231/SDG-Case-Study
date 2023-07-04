################################################
################################################
################################################
# advanced analysis
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
import pickle
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve


################################################
# read data
################################################

# Define the name of the CSV file
file_name = "SDG-Case-Study//dataset.csv"

# Read the CSV file from the mounted volume using Pandas
df = pd.read_csv(file_name,
                 delimiter=';',
                 decimal=',',
                 error_bad_lines=False)
print('data imported')
################################################
# data tranformation
################################################

# save the customer_id as index, as it's not needed for analysis
df.index = df['Customer_ID']
df = df.drop('Customer_ID', axis=1)
print('Customer_ID saved as index')


# prepare data with one-hot encoding


# Select the categorical columns to be encoded
cat_columns = [
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

# Perform one-hot encoding on the selected columns
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(df[cat_columns])

# Convert the encoded data into a pandas dataframe and concatenate it with the original dataframe
onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), 
                                 columns=onehot_encoder.get_feature_names(cat_columns),
                                 index= df.index)
df = df.merge(onehot_encoded_df, right_index=True, left_index=True)

# Drop the original categorical columns that were encoded
df = df.drop(cat_columns, axis=1)

#################################################################################

# numerical columns: 

num_columns = [
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

# change the data type of the selected columns
df[num_columns] = df[num_columns].astype(float)

###################################################################

# replace NaN values with the mean of the variable
df = df.fillna(0)

# replace inf values with a large constant value
df = df.replace([np.inf, -np.inf], 1e10)

###################################################################
# Split the data into features (X) and target (y)
X = df.drop('churn', axis=1)
y = df['churn']

print(X.shape)
# Save processed data:
X.to_csv('SDG-Case-Study//dataset_processed_X.csv')
y.to_csv('SDG-Case-Study//dataset_processed_y.csv')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('data splited into training and testing sets')

# Create model
model = RandomForestClassifier(random_state=42)
print('created a random forest classifier model')

# Train the model on the training data
model.fit(X_train, y_train)
print('trained model')

# Save the trained model as a pickle file
with open('SDG-Case-Study//output//dt_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('saved model as pickle in output folder')


# Make predictions on the testing data
y_pred = model.predict(X_test)
print('predictions made on testing data')

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate the precision of the model
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Calculate the recall of the model
recall = recall_score(y_test, y_pred)
print('recall:', recall)

# Calculate the f1 of the model
f1 = f1_score(y_test, y_pred)
print('f1:', recall)

# Plot and save the ROC curve
roc_plot = plot_roc_curve(model, X_test, y_test)
plt.savefig('SDG-Case-Study//plots//dag02_roc_curve.png')

# Get the feature importances from the model
importances = model.feature_importances_

# Print and save the feature importances
importances_df = pd.DataFrame(columns = ['Feature', 'Importance'])
for i, importance in enumerate(importances):
    row = {'Feature': X.columns[i], 'Importance': importance}
    importances_df = importances_df.append(row, ignore_index=True)

importances_df = importances_df.sort_values(by='Importance',
                                            ascending=False)
importances_df.to_csv('SDG-Case-Study//output//dag02_importances.csv',index=False)

print('Most important features: \n ', importances_df[0:20])

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
    plt.savefig(f"SDG-Case-Study//plots//{feature_name}_ice_plot.png")
    plt.close()

sys.exit(0) # exit with success status code