import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# recommendation systems are of two types-
# 1. Content based recommendation system (Item-based filtering)
# 2. Collaborative filtering based recommendation system
# Here we will be making a Content based recommendation system (Item-based filtering)

# Reading u.data which contains information about user ratings
# Also since our CSV does not has any column names we will pass it on our own
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("./ml-100k/u.data", sep='\t', names=column_names)
print(df)

# Since the above dataframe does not contains movie titles we will extract it from
# other dataset (u.item) which contains info about the movies
movie_titles_data = pd.read_csv('ml-100k/u.item', sep='|', header=None)
print(movie_titles_data)
# Now extracting item_id and corresponding movie title from the dataframe
movie_titles = movie_titles_data.iloc[:, [0, 1]]
print(movie_titles)
# Giving column names to movie_titles dataframe
movie_titles.columns = ['item_id', 'title']
print(movie_titles)

# Merging movie title dataframe and ratings dataframe
df = pd.merge(df, movie_titles, on='item_id')
print(df)

# Doing some exploratory data analysis
sns.set_style('white')

# Finding the average rating of a movie
# First grouping movies by title and then finding mean of the columns
# and finally taking only ratings columns and sorting it in descending order
average_rating = df.groupby('title').mean(
)['rating'].sort_values(ascending=False)
print(average_rating)

# But when we see the above dataframe we also realize that not only just the ratings
# are important but we also need the number of ratings as if a movie is 5 star rated but
# has been rated only by one user then that is not necessarily better than other movies with
# rating less than 5 stars hence to get a more accurate overview we also need to have no of ratings
# So creating a dataframe of average rating and number of ratings
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['number_of_ratings'] = pd.DataFrame(
    df.groupby('title').count()['rating'])
print(ratings)

# Sorting the ratings dataframe according to rating just to explore the highest and lowest
# rated films
print(ratings.sort_values(by='rating', ascending=False))

# Plotting a histogram of number of ratings to see how many movies have less people rating them
# The following is a plot of the number of ratings on X-axis and number of movies having that many
# ratings on Y-axis
plt.figure(figsize=(10, 6))
plt.hist(ratings['number_of_ratings'], bins=70)
plt.show()

# Plotting a histogram of ratings vs the number of movies having that rating
plt.hist(ratings['rating'], bins=70)
plt.show()
# From this plot we observe that the data resembles normal distribution as we get a bell shaped curve

# Plotting a joint plot of ratings and number of ratings
sns.jointplot(x='rating', y='number_of_ratings', data=ratings, alpha=0.5)
plt.show()
# This plot shows us that as the rating increases so does the number of ratings except for
# some exceptions

########
# Creating Movie Recommendation system
########

# Creating a table of users(as row indices) and movies(as column indices) with each
# datapoint depicting rating of that movies by that user
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat)

# Creating the function to predict movies/recommend movies based on the
# current movie user has watched


def predict_movies(movie_name):
    # Getting the individual user rating of that movie
    movie_user_ratings = moviemat[movie_name]
    # Correlating other movies from moviemat with the current movie user ratings
    # The way correlation works here while correlating pandas series is that movies
    # having max number of same users giving them the same rating are highly correlated
    # and vice versa
    # Getting this correlation helps relate which movies are more similar
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    # similar_to_movie is a pandas series , so converting it into pandas dataframe
    corr_to_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    # Dropping the rows having NaN as correlation
    # meaning those movies had 0 common users rating there movies
    # Dropping them as they would cause problems as since they are not a number we can't operate
    # on them or compare them and anyways they are useless as they have NaN correlation so they are
    # not a good recommendation anyhow
    corr_to_movie.dropna(inplace=True)
    # Joining number of ratings column from ratings dataframe to corr_to_movie
    # We are joining that here as , as we saw earlier while exploring there were a lot of movies with
    # low number of ratings and since they have low users rating them we have little data on them and
    # predicting/recommending such movies won't be considered a good practice
    corr_to_movie = corr_to_movie.join(ratings['number_of_ratings'])
    # Sorting the corr_to_movie dataframe, where number of ratings more than atleast 100, according to
    # the correlation to get the best recommendations
    predictions = corr_to_movie[corr_to_movie['number_of_ratings'] > 100].sort_values(
        by='Correlation', ascending=False)
    # Finally returning predictions dataframe as our answer
    return predictions


# Checking if our recommendation system works
# We can check by simply recommending for a few movies and seeing if those recommendations are good or not
predictions = predict_movies('Titanic (1997)')
# Taking only the top few as those are the best recommendations
print(predictions.head())
predictions = predict_movies('Toy Story (1995)')
# Taking only the top few as those are the best recommendations
print(predictions.head())
predictions = predict_movies('Star Wars (1977)')
# Taking only the top few as those are the best recommendations
print(predictions.head())
