import sys
import numpy as np
from datetime import datetime

# np.arange can be used to create an array to create an array
# containing a given range like range() function in python
arr = np.arange(10000000)

# Using numpy is better than using normal arrays as can be seen in time difference for below operation
# on a normal array vs a numpy array
pythlist = list(range(10000000))
print(datetime.now().strftime("%H:%M:%S.%f"))
arr *= 3
print(datetime.now().strftime("%H:%M:%S.%f"))
print(datetime.now().strftime("%H:%M:%S.%f"))
pythlist = [item*3 for item in pythlist]
print(datetime.now().strftime("%H:%M:%S.%f"))

# To create an array in numpy we use np.array
np_array = np.array([[1, 2, 3, 6, 7, 8], [3, 5, 6, 7, 8, 0]])
print(np_array)
print(type(np_array))
print(np_array.shape)
print(np_array.dtype)

# np.zeros creates an array of zeros
np_array = np.zeros(4)
print(np_array)
np_array = np.zeros((4, 6))
print(np_array)
print(np_array.dtype)

# np.ones creates an arrays of ones
np_array = np.ones(4)
print(np_array)
print(np_array.dtype)

# Creating an array of constants - np.full((rows, columns), constant)
np_array = np.full((3, 2), 5)
print(np_array)

# np.empty creates an array of garbage values while np.zeros creates an array of zeros
np_array = np.empty((4, 6))
print(np_array)

# Operations on numpy arrays
array1 = np.arange(5)
array2 = np.array([5, 6, 7, 8, 9])
print(array1+array2)
print(array1-array2)
print(array1*array2)
print(array1/array2)
print(array1**array2)
print(1/array2)

# Manually defining datatypes for a numpy array
np_array = np.array([1, 2, 3, 4], np.int32)
print(np_array)
print(np_array.dtype)

# Accessing elements in a numpy array
np_array = np.array([[1, 2, 3, 6, 7, 8], [3, 5, 6, 7, 8, 0]])
print(np_array.shape)
print(np_array[0])
print(np_array[0, 1])
print(np_array[0, 2])

# np.linspace(start,end,number_of_elements) creates equally spaced array between the given start and end
# values and the number of elements is equal to the provided value
print(np.linspace(1, 4, 4))

# np.identity(n) creates a n X n identity matrix
print(np.identity(10))

# array.reshape(rows,columns) reshapes the given matrix into the given dimensions if possible else throws an
# error
np_array = np.arange(99)
np_array = np_array.reshape(3, 33)
print(np_array)
# np_array.reshape(3, 31)  # This line throws an error

# array.ravel() convert the given array into a 1D array
np_array = np_array.ravel()
print(np_array)

# Number of axis in an array is the number of dimensions of the array
# 1D array - 1 Axis - [Axis0]
# 2D array - 2 Axis - [Axis0, Axis1] Here Axis0 depicts Y-axis and Axis1 depicts X-axis
# 3D array - 3 Axis - [Axis0, Axis1, Axis2]
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 1, 0]])
print(arr)
print(arr.sum(axis=0))  # Prints sum of all columns
print(arr.sum(axis=1))  # Prints sum of all rows
print(arr.T)  # Prints transpose of the array
for i in arr.flat:  # Here array.flat helps traverse the array as if it were a 1D array
    print(i)
print(arr.ndim)  # Prints the dimension of array
print(arr.size)  # Prints the size of array
print(arr.nbytes)  # Prints the total bytes consumed by the elements of the array

# Some operations for 1D array
# Also note that 1D arrays are called vectors
arr = np.array([1, 2, 3, 4, 5])
print(arr.argmax())  # Prints the index of occurence of the max element in the array
print(arr.argmin())  # Prints the index of occurence of the min element in the array
print(arr.argsort())  # Prints the index of elements of the array in sorted order
# If we perform th eabove operations on a 2D array then numpy would first convert them into a 1D array and
# then perform the various operations except argsort which works normally for a 2D array
# But if we provide it with the argument axis=n then it return index of occurence of max element of each
# column/row
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 1, 0]])
print(arr.argmax())
print(arr.argmin())
print(arr.argsort())
print(arr.argmax(axis=0))
print(arr.argmin(axis=1))
print(arr.argsort(axis=0))

# Element wise operations using numpy
arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 1, 0]])
arr2 = np.array([[1, 2, 1], [4, 0, 6], [8, 1, 0]])
print(arr1+arr2)  # Prints the matrix sum of the respective elements of the matrices
# Prints the matrix multiplication of the respective elements of the matrices
print(arr1*arr2)
print(np.sqrt(arr1))  # Prints the squareroot of each element of arr1
print(arr1.sum())  # Prints sum of all elements of arr1
print(arr1.max())  # Prints max element of arr1
print(arr1.min())  # Prints min element of arr1

# np.where() can be used to search in array
print(np.where(arr1 > 5))  # Prints locations of elements in arr1 greater than 5

# np.count_nonzero(array) returns the count of non zero elements in the array
print(np.count_nonzero(arr1))

# numpy arrays take less space than normal python lists
py_arr = [1, 2, 3, 4, 5]
np_arr = np.array(py_arr)
print(sys.getsizeof(1)*len(py_arr))  # Prints space consumed by python lists
print(np_arr.itemsize*np_arr.size)  # Prints space consumed by numpy array
# We observe numpy arrays take considerably less space than python lists

# array.to_list() converts numpy array into python lists
np_arr = np_arr.tolist()
print(type(np_arr))

# Accessing/Updating elements of the matrix - array[start:end+1,start:end+1]
# or array[start:end+1, column] or array[row,start:end+1]
np_array = np.array([[1, 2], [3, 4], [5, 6]])
print(np_array[:, 1])
np_array[:, 1] = [1, 2, 3]
print(np_array)
np_array[:, 1] = 5
print(np_array)

# Matrix multiplication/Dot product
arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 1, 0]])
arr2 = np.array([[1, 2, 1], [4, 0, 6], [8, 1, 0]])
print(arr1.dot(arr2))
print(np.dot(arr1, arr2))  # Gives the same result as above statement

# Vector dot product of two vectors
v1 = np.array([1, 2, 3, 4])
v2 = np.array([1, 2, 3, 4])
print(v1.dot(v2))

# Stacking two arrays into one - np.stack((array1, array2), axis = zero_or_one)
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])
print(np.stack((v1, v2), axis=0))
print(np.stack((v1, v2), axis=1))

# Exploring the np.random module
# Shuffling an array
np_arr = np.arange(5)
np.random.shuffle(np_arr)
print(np_arr)
# Creating a random matrix - np.random.random((rows,columns)) or np.random.rand(rows,columns)
np_array = np.random.random((3, 2))
print(np_array)
np_array = np.random.rand(3, 2)
print(np_array)
# Return a sample (or samples) from the “standard normal” distribution - np.random.randn(rows, columns)
print(np.random.randn())
print(np.random.randn(3, 2))
# Give a number of random inetgers between given values - np.random.randint(start,end+1,number_of_integers)
print(np.random.randint(5, 12, 3))
# Randomly pick an element from an array
print(np.random.choice(np_arr))

# Statistical operations using numpy
# Mean of an array
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = np.array([9, 10, 11, 12])
print(np.mean(b))
print(np.mean(a))
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))
# Median of array
a = np.array([1, 5, 4, 2, 0])
b = np.array([1, 2, 3, 4, 5, 6])
c = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(np.median(a))
print(np.median(b))
print(np.median(c))
# Weighted Average of an array (meaning each element has a weight associated with it)
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
print(np.average(a))
print(np.average(a, weights=b))
