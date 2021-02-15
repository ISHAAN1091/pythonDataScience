import numpy as np
import pandas as pd

# Creating some data in the form of a python dictionary
user_data = {
    "MarksA": np.random.randint(1, 100, 6),
    "MarksB": np.random.randint(1, 100, 6),
    "MarksC": np.random.randint(1, 100, 6)
}

# Converting the above dictionary into a pandas dataframe
# dtype - used to manually define the datatype to be used
df = pd.DataFrame(user_data, dtype='float32')
print(df)

# Use the head method to print the data frame
# n - used to define how many rows to print , has default value of 5
print(df.head(n=3))

# To see all column titles use columns attribute
print(df.columns)

# Converting a dataframe to a CSV file
df.to_csv('marks.csv')

# Reading from a CSV file and getting pandas dataframe
my_data = pd.read_csv('marks.csv')
print(my_data)

# Deleting columns from dataframe
my_data = my_data.drop(columns=['Unnamed: 0'])
print(my_data)

# Use describe method to get info about the dataframe like mean, standard deviation,etc
print(my_data.describe())

# Use the tail method to print data frame from bottom
# n - used to define how many rows to print from top, has default value of 5
print(df.tail(n=3))

# Accessing elements in the dataframe
# Accessing a particular row in the dataframe
print(df.iloc[3])
# Accessing a particular element
print(df.iloc[3][1])
print(df.iloc[3, 1])

# Searching for index of a column from its title
index = df.columns.get_loc("MarksB")
print(index)

# Accessing multiple datapoints
index = [df.columns.get_loc("MarksB"), df.columns.get_loc("MarksC")]
print(df.iloc[3, index])
print(df.iloc[:3, index])

# Sorting your dataframe
# ascending - used to define whether you want to arrange in ascending order or descending order
# by - used to define to sort using which parameters and in what priority order
df = df.sort_values(by=['MarksA'], ascending=True)
print(df)
# Giving more than one values in the by list
# Here it will first sort according to MarksA and then take into consideration MarksC
# if at some point we have same value for MarksA
df = df.sort_values(by=['MarksA', 'MarksC'], ascending=True)
print(df)

# Converting dataframe into a numpy array
df_array = df.values
print(type(df_array))
print(df_array)

# Converting numpy arrays into dataframes
new_df = pd.DataFrame(data=df_array, dtype='int32', columns=[
                      'Physics', 'Chemistry', 'Maths'])
print(new_df)
