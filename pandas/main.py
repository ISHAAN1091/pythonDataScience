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
# index - used to manually define what to use as indices, default value is range indices
df = pd.DataFrame(user_data, dtype='float32')
print(df)
# Creating dataframe from random values
newdf = pd.DataFrame(np.random.rand(334, 5), index=np.arange(334))
print(newdf)

# Use the head method to print the data frame
# n - used to define how many rows to print , has default value of 5
print(df.head(n=3))
print(df.head(3))  # Another way of writing the above statement

# To see all column titles use columns attribute
print(df.columns)

# To see all row labels/indices use .index attribute
print(df.index)

# Converting a dataframe to a CSV file
# index - used to tell python whether or not to include the indices column in dataframe or not
df.to_csv('marks.csv')
df.to_csv('marks_index_false.csv', index=False)

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
print(df.tail(3))  # Another way of writing the above statement

# Accessing elements in the dataframe
# The thing about iloc is that it doesn't care for the labels it just counts the indices
# So you could have A,B,C,etc on row/column labels but iloc would still use .iloc[3,1] as
# iloc only cares for the indexing
# Accessing particular column in the dataframe
print(df["MarksA"])
# Accessing a particular row in the dataframe
print(df.iloc[3])
# Accessing a particular element through dataframe[column][row]
print(df["MarksA"][1])
# Accessing a particular element through dataframe.iloc
print(df.iloc[3][1])
print(df.iloc[3, 1])
# Gives a table of elements of row1 and row2 for the columns0,1,2
print(newdf.iloc[[1, 2], [0, 1, 2]])
# Gives a table of elements of all rows for the columns0,1,2
print(newdf.iloc[:, [0, 1, 2]])
# if you don't want to access elements or rows using iloc or using their indices
# meaning you want to access them using their labels then use loc-
print(df.loc[1, "MarksA"])

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

# Series in pandas
# Series is a one-dimensional labeled array capable of holding data of one type.
ser = pd.Series(np.random.rand(34))
print(ser)
print(type(ser))

# Creating dataframe from random values
newdf = pd.DataFrame(np.random.rand(
    334, 5), index=np.array([x+20 for x in range(334)]))
print(newdf)

# Sorting a dataframe according to column/row labels
newdf = newdf.sort_index(axis=0, ascending=False)
print(newdf)
newdf = newdf.sort_index(axis=1, ascending=False)
print(newdf)

# Creating a copy of a dataframe
newdf_copy = newdf.copy()
print(newdf_copy)

# Changing column labels of a dataframe
newdf.columns = list('ABCDE')
print(newdf)

# Changing row labels of a dataframe
newdf.index = np.arange(334)
print(newdf)

# Updating values in a dataframe using dataframe.loc[row,column]=new_value
# You can use either loc or iloc for updating values
newdf.loc[0, 'A'] = 650
newdf.iloc[1, 0] = 600
print(newdf)

# Deleting columns/rows from dataframe
# axis - Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’), default value 0.
# index - single label or list-like ,Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).
# columns - single label or list-like ,Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
# inplace - used to tell whether to modify in main dataframe or not
# Deleting rows
newdf = newdf.drop(0)  # Deletes row with label 0
print(newdf)
newdf = newdf.drop(index=1)  # Deletes row with label 1
print(newdf)
# Deleting columns
newdf = newdf.drop('E', axis=1)  # Deletes column with label 'E'
print(newdf)
# Deletes column with labels 'C' and 'D'
newdf = newdf.drop(columns=['C', 'D'])
print(newdf)
# Deletes the rows 2 and 3 in newdf permanently, without reassigning like above
newdf.drop([2, 3], inplace=True)
print(newdf)

# Searching in dataframe
# Also note that here in searching we cannot use 'and' or 'or' operators (logical operators) with pandas series
# Here we will have to use bitwise operators
print(newdf.loc[(newdf['A'] < 0.3) & (newdf['B'] > 0.1)])

# Resetting index
# Since we deleted some rows above it has upset our index pattern and we want to reset the pattern
# we can do this
# drop - used to tell to not add a new row of indices and change in the current index
# inplace - used to tell whether to modify in main dataframe or not
newdf.reset_index(drop=True, inplace=True)
print(newdf)

# .isnull() method to find null datapoints
newdf.loc[1, 'B'] = None
print(newdf['B'].isnull())
# Printing out indices with null data points
for index, i in enumerate(newdf['B'].isnull()):
    if i:
        print(f'Null value found at {index}')

# Creating some data in the form of a python dictionary
df = pd.DataFrame({"name": ["Alfred", "Batman", "Catwoman"],
                   "toy": [np.nan, "Batmobile", "Bullwhip"],
                   "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT],
                   "nullRow": [np.nan, pd.NA, np.nan]})
print(df.head())

# Dropping rows/column with null elements
print(df.dropna())  # Drops a row even if a single element in it is NA
print(df.dropna(how='all'))  # Drops a row only if all elements in it is NA
print(df.dropna(axis=1))  # Drops a column even if a single element in it is NA
# Drops a column only if all elements in it is NA
print(df.dropna(how='all', axis=1))

# You can use drop_duplicates method to delete duplicate values from rows/columns


# Some statistical methods in pandas
# df.mean()
# df.corr()
# df.count()
# df.max()
# df.min()
# df.median()
# df.std()

# You can also handle excel sheets with pandas
# .read_excel() method and other methods are inbuilt as well for excel
