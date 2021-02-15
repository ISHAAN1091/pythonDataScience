import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In this problem we plot a graph with number of character in title name on X-axis
# and number of movies with that many character in their name on Y-axis
# Read a CSV
df = pd.read_csv("./movie_metadata.csv")
# Getting all the titles of the movies from the column movie_title
titles = list(df.get('movie_title'))
# Cleaning data
for index, title in enumerate(titles):
    if '\xa0' in title:
        titles[index] = title[:title.find('\xa0')]
# Finding the number of character vs number of character
# in their title info from the above list
freq_titles = {}
for title in titles:
    length = len(title)
    if freq_titles.get(length) is None:
        freq_titles[length] = 1
    else:
        freq_titles[length] += 1
# Getting X and Y coordinates to make the plot
X = np.array(list(freq_titles.keys()))
Y = np.array(list(freq_titles.values()))
# Plotting a scatter plot to visualise this data
plt.scatter(X, Y)
plt.xlabel("Length of Movie Titles")
plt.ylabel("Number of Movies")
plt.title("Movie Data Visualization Problem")
plt.show()
# We can see from the scatter plot that we get a bell shaped curve and hence a normal distribution
