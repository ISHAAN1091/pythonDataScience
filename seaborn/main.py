import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Loading datasets from seaborn library
tips = sns.load_dataset('tips')

# Plotting a bargraph
# x/y - used to define what column from data to plot on X-axis/Y-axis
# data - used to define the data from which plot is to be made
# estimator - used to define the function to calculate the height of the bar (default value is mean)
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
plt.show()
# Plots created using seaborn need to be displayed like ordinary matplotlib plots.
# This can be done using the plt.show()

# Plotting a countplot
# A count plot simply just plots the counts or the frequency on Y-axis and
# hecnce we only define what we want on X-axis
# x - used to define what column from data to plot on X-axis
# data - used to define the data from which plot is to be made
sns.countplot(x='sex', data=tips)
plt.show()

# Plotting boxplot
# Boxplot kind of shows the normal distribution
# It shows the global minimum and maximum of normal distribution
# It also denotes mean(the central line in colored area)
# and mean+sigma to mean-sigma (this is represented by the colored area)
# where sigma is standard deviation and the points/dots marked on the graph are outliers
# Outliers are the data points not falling within the range are like exception
# x/y - used to define what column from data to plot on X-axis/Y-axis
# data - used to define the data from which plot is to be made
# hue - used to plot graph of x vs y w.r.t another variable as well
sns.boxplot(x='day', y='total_bill', data=tips, hue='sex')
plt.show()

# Plotting violinplot
# It is just another representation of boxplot. Instead of the colored box
# we have a thick center line and instead of a line to mark max and min we have the violin
# shaped colored part with the outliers included at its edges in the colored part
# x/y - used to define what column from data to plot on X-axis/Y-axis
# data - used to define the data from which plot is to be made
# hue - used to plot graph of x vs y w.r.t another variable as well
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
plt.show()
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex')
plt.show()

# Finding correlation
tips_corr = tips.corr()
print(tips_corr)

# Plotting a heatmap
# Heatmap is for visualising dataframes with variables in both rows and columns
# like correlation table
# Heatmap visualises by showing higher value with color of stringer intensity(darker color)
# and lower value by lighter color or color of lower intensity as can be seen in the case below
# annot - used to define whether or not to display the respective values of section in heatmap
# cmap - used to change the color theme to be used in heat map
sns.heatmap(tips_corr, annot=True, cmap="coolwarm")
plt.show()

# Loading datasets from seaborn library
flights = sns.load_dataset('flights')

# Getting a pivot_table from flights dataset (this is a pandas feature)
flights_table = flights.pivot_table(
    index='month', columns='year', values='passengers')
print(flights_table)

# Plotting the heatmap of flights_table
sns.heatmap(flights_table, cmap="coolwarm")
