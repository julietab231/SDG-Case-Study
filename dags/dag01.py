################################################
################################################
################################################
# descriptive analysis
################################################
################################################
################################################


################################################
# install libraries
################################################
import pandas as pd

################################################
# read data
################################################

# Define the name of the CSV file
file_name = "SDG-Case-Study//dataset.csv"

# Read the CSV file from the mounted volume using Pandas
df = pd.read_csv(file_name,
                 delimiter=';',
                 error_bad_lines=False)

################################################
# data analysis
################################################

# Print head of data
print('Head:\n',df.head())

# check the shape of the DataFrame
print('\nShape of the DataFrame:\n', df.shape)  

# check the data types of each column
print('\nData types of each column:\n', df.dtypes)

# get summary statistics of the numerical/date format column
print('\nSummary statistics \n',df.describe()) 

# count the number of unique values in each column
print('\nUnique values in each column:\n', df.nunique())

# check the frequency distribution of the 'churn' column
print('\nDistribution of the churn column:\n', df['churn'].value_counts())
# churn takes values from 0 to 1
# mean of churn is 0.49
# Distribution of the churn column:
# 0    50438 -- 50%
# 1    49562 -- 50%

