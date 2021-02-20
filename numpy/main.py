import matplotlib.pyplot as plt
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
# Below element wise matrix multiplication is also known as "Hadamard product"
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
# Standard Deviation array
print(np.std(a))
# Variance of an array
print(np.var(c))

# Tensors - Tensors are arrays with more than two axis
t = np.zeros((5, 5, 3))
print(t)
# Images are also stored as tensors as they are a 2D collection of pixels
# and a pixel is an array of RGB values hence an image is stored like a tensor
# Visualizing an image from tensor
t = np.zeros((5, 5, 3), dtype='uint8')
# Here we used uint8 i.e. unsigned int 8 as images don't need more space
# than that hence to make it more space efficient we did so
plt.imshow(t)
plt.show()
# Transpose of a tensor
t = np.zeros((50, 25, 3))
print(t.shape)
# Without any other information just reverses the dimensions
t1 = np.transpose(t)
print(t1.shape)
# With the axes parameters it changes the dimensions accordingly in the provided order
t2 = np.transpose(t, axes=(2, 0, 1))
print(t2.shape)

# Broadcasting -
# You can add scalar to vector and numpy will add it to each element of the vector
a = np.array([1, 2, 3, 4])
print(a+5)
# Similarly you can add vector to matrix and numpy will add it to each column of the matrix
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(b+a)

# Norm of a vector
# Norm can be thought of as a proxy for size of a vector
# we define pth norm of a vector x as - ((summation((absolute value of i-th element of x)^p))^1/p)
x = np.array([-3, 4])
lp2 = np.linalg.norm(x)
print(lp2)
lp1 = np.linalg.norm(x, ord=1)
print(lp1)
# Calculating infinity norm - Infinity norm return the absolute value of the largest element of the array
lpinf = np.linalg.norm(x, ord=np.inf)
print(lpinf)

# Determinant of a matrix
a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))

# Inverse of a matrix
inv = np.linalg.inv(a)
print(inv)
print(inv.dot(a))

# Pseudo inverse of a matrix -
# If a matrix does not has an inverse it would still have a pseudo inverse always
a = np.array([[1, 1], [1, 1]])
pinv = np.linalg.pinv(a)
print(pinv)
print(np.dot(a, pinv))
# Here if the matrix is invertable then pseudo inverse is same as inverse
# Hence while solving a system of linear equations using matrices it is preferable to use pseudo inverse
# rather than normal inverse as we can entirely avoid the problem of noninvertable matrices

# Solving a system of linear equations
a = np.array([[2, 3], [3, 1]])
b = np.array([8, 5])
sol = np.linalg.solve(a, b)
print(sol)

# Getting a Univariate Standard Normal Distribution
# Suppose we need 100 values of a standard normal dist. and it should be around mean=60 and sigma = 5
Xsn = np.random.randn(100)
print(Xsn)
sigma1 = 5
u1 = 60
X1 = Xsn*sigma1+u1
print(X1)
# Suppose we need 100 values of a standard normal dist. and it should be around mean=40 and sigma = 5
sigma2 = 5
u2 = 40
X2 = Xsn*sigma2+u2
print(X2)
# We observe that for a normal distribution mean is the central value around/near which we have most of the
# data and sigma represents the range of spread of the values around the mean
# For example in the first case above with u=60 and sigma=5 most of the values will lie in the range
# 60+5 and 60-5 i.e. from 55 to 65

# Generating a Bivariate Normal Gaussian Distribution
# Suppose we need 500 values with a mean u and covariance cov
# Also a bivariate normal distribution is shaped like a bell shaped curve but in both x and y directions
# The height and ellipticalness of this bell curve depends on the covariance between x and y i.e. cov-xy
# or cov-yx where cov is the 2X2 matrix of [[cov-xx cov-xy], [cov-yx cov-yy]]
# Example 1
mean = np.array([0.0, 0.0])
cov = np.array([[1, 0.0], [0.0, 1]])
distbn = np.random.multivariate_normal(mean, cov, 500)
print(distbn)
plt.scatter(distbn[:, 0], distbn[:, 1])
plt.show()
# Example 2
mean = np.array([0.0, 0.0])
cov = np.array([[1, 0.5], [0.5, 1]])
distbn = np.random.multivariate_normal(mean, cov, 500)
print(distbn)
plt.scatter(distbn[:, 0], distbn[:, 1])
plt.show()
# Example 3
mean = np.array([0.0, 0.0])
cov = np.array([[1, 0.8], [0.8, 1]])
distbn = np.random.multivariate_normal(mean, cov, 500)
print(distbn)
plt.scatter(distbn[:, 0], distbn[:, 1])
plt.show()
# Example 4
mean = np.array([0.0, 0.0])
cov = np.array([[2, -0.8], [-0.8, 1]])
distbn = np.random.multivariate_normal(mean, cov, 500)
print(distbn)
plt.scatter(distbn[:, 0], distbn[:, 1])
plt.show()
# Notice how the curve becomes more and more elliptical as the value of cov-xy/cov-yx(both are the same)
# increases . This is because as they increase that means the dependance between x and y becomes stronger
# Also cov-xx cov-yy represent the spread/thickness of the plot in x and y directions
