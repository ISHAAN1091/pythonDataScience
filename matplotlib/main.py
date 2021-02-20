import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y1 = x**2
y2 = 2*x+3
print(x)
print(y1)
print(y2)

# Available plot themes in matlpotlib
themes = plt.style.available
print(themes)
# Changing theme of plot
plt.style.use('seaborn')

# Plotting graphs
plt.plot(x, y1, color='red')
plt.plot(x, y2, color='green')
plt.show()
# Also note that plt.show() marks the ending of one graph and the next plots
# will be made on a new graph
plt.plot(x, y2, color='green')
plt.show()

# Applying labels, title, legend, linestyle, marking points
# color - used in plot() to manually define the color of the plot
# label - used to give legend to line
# marker - used to mark points with a defined symbol
# linestyle - used to give line a style like dashed, etc
# xlabel/ylabel - used to assign labels to respective axes
# xlim/ylim - used to define the range of values on y axis and x axis
# .legend() - used to display legends
# .title() - used to give title to the graph
plt.plot(x, y1, color='red', label="Apple", marker='o')
plt.plot(x, y2, color='green', label="Kiwi", linestyle="dashed", marker='*')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Prices of fruits overtime")
plt.ylim(0, 40)
plt.xlim(0, 10)
plt.legend()
plt.show()

# plotting an array
prices = np.array([1, 2, 3, 4])**2
print(prices)
# You can use plt.figure(figsize=(horizontal_size,vertical_size)) to resize the plot proportianately
plt.figure(figsize=(2, 2))
plt.plot(prices)
plt.show()
# In the case where only one array is available those are automatically assumed to be
# the y co-ordinates and the indices of each value from the array are assumed to be
# x co-ordinate

# Scatter plots
plt.scatter(x, y1)
plt.scatter(x, y2)
plt.show()
# All the styling properties like label, marker, etc defined above are also valid here

# Bar Graphs
# plt.bar(x_coordinates, array_of_heights) is used to plot bar graph
# width - used to define the width of the bars of the plot
# label - used to give legend to the  bars plot
# tick_label - used to assign individual xlabels to each bar of the plot
# color - used in plot() to manually define the color of the plot
# xlabel/ylabel - used to assign labels to respective axes
# xlim/ylim - used to define the range of values on y axis and x axis
# .legend() - used to display legends
# .title() - used to give title to the graph
x_coordinates = np.array([0, 1, 2])*2
plt.bar(x_coordinates-0.25, [10, 20, 15], width=0.5,
        label="Current Year", tick_label=["Gold", "Platinum", "Silver"])
plt.bar(x_coordinates+0.25, [20, 10, 12],
        width=0.5, label="Next Year", color="red")
plt.title("Metal Price Comparision")
plt.xlabel("Metal")
plt.ylabel("Price")
plt.ylim(0, 40)
plt.xlim(0, 10)
plt.legend()
plt.show()

# Pie Charts
# plt.pie(proportion_of_various_part_of_pie_chart)
# labels - used to assign labels to various parts of the pie chart
# explode - used to define the distance of the piece/part from the centre of the pie
# autopct - used to specify whether to print respective percents and their format
# shadow - used to give shadows to pieces of the pie
# .title() - used to give title to the graph
subjects = ["Maths", "English", "Science", "Social Studies"]
weightage = [20, 10, 15, 5]
plt.pie(weightage, labels=subjects, explode=(
    0, 0, 0.1, 0), autopct='%1.1f%%', shadow=True)
plt.title("Subjects")
plt.show()

# Histograms
# alpha - used to define the opacity of the histogram plot
# labels - used to assign labels to various parts of the pie chart
# xlabel/ylabel - used to assign labels to respective axes
# .legend() - used to display legends
# .title() - used to give title to the graph
plt.hist(X1, alpha=0.5, label="Maths")
plt.hist(X2, label="Physics")
plt.ylabel("Prob/FreqCount of Students")
plt.xlabel("Marks Range")
plt.title("Histogram")
plt.legend()
plt.show()
