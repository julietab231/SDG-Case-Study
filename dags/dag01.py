########################
# install libraries
########################
!pip install pandas
import pandas as pd


########################
# import data
########################

# Define the path to the folder containing the CSV file

# Define the name of the CSV file
file_name = "dataset.csv"

# Read the CSV file from the mounted volume using Pandas
df = pd.read_csv(file_name)

# Do something with the data (e.g. print the first 5 rows)
print(df.head())