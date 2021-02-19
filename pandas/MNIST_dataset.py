import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading the MNIST dataset
# MNIST dataset is a data of greyscale images of digits
# images are stored as pixels and since it is a greyscale pixel it only has
# one information about it i.e. its intensity
# In the MNIST datatset the label column tells the digit the image is of
# and rest are the pixels of the image
df = pd.read_csv('./pandas/mnist_train.csv')
print(df)

# Converting the dataframe into a numpy array
data = df.values
# Also we are going to randomly shuffle the data array as we don't want their to be any pre
# existing order or arrangement in dataset that might have come from the source of the data
# This only shuffles the rows and not the inner elements of the rows thereby keeping our data intact
np.random.shuffle(data)
print(data)
print(type(data))

# Separating the 'labels' column and the pixels columns in the numpy array
# We store labels in Y and pixels columns in Y
X = data[:, 1:]
print(X)
Y = data[:, 0]
print(Y)

# Visualizing the images
# Here while plotting the image we reshaped the pixels again into a square
# image and defined it to be a grayscale image


def drawImg(X, Y, i):
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title("Label "+str(Y[i]))
    plt.show()


# Visualising the first 5 images
for i in range(5):
    drawImg(X, Y, i)

# Splitting the dataset
# While working on Machine Learning algorithms we generally do not use our entire data
# to train our model , we split it in two parts one is used for testing and the other is
# used for training
split_percent = 0.80  # Splitting 80% of data for training
split = int(split_percent*X.shape[0])
print(split)
X_train, Y_train = X[:split, :], Y[:split]
X_test, Y_test = X[split:, :], Y[split:]
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Plotting a visualization of the first 25 images
# We will be plotting it in a grid of 25 subplots
# For this we will use the subplot feature of matplotlib
# which is basically a grid of plots
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(Y_train[i])
    plt.axis('off')
plt.show()

# Using sklearn for data splitting
# Also if we  don't want to split manually we can use sklearn library to do it for us
# we will use train_test_split
# test_size - used to define the what percentage of total data should be test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
