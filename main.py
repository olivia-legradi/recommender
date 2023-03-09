import numpy as np
from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate
import pandas as pd

import random

df = pd.read_csv("ml-latest-small/ratings.csv")
df.drop("timestamp", inplace=True, axis=1)
moviesDf = pd.read_csv("ml-latest-small/movies.csv")
allMovieIds = set(pd.Series(moviesDf.movieId).unique())
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df, reader)
algo = SVD()
absoluteMovieRatings = pd.DataFrame(index=pd.Index(allMovieIds))
relativeMovieRatings = pd.DataFrame(index=pd.Index(allMovieIds))
performance = pd.DataFrame(index=pd.Index(['RMSE']))

numberOfUsers = 2
numberOfIterations = 6
trainingStep = 2
evaluationStep = 2
measuringStep = 2

#trains the algorithm
def train_algorithm():
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

# selecs a number of random users from the existing ones
def select_users():
    selectedUsers = set()
    allUsers = np.array(pd.Series(df.userId).unique())
    for i in range(numberOfUsers):
        while True:
            randomNumber = random.randint(0, len(allUsers)-1)
            randomUser = allUsers[randomNumber]
            if not randomUser in selectedUsers:
                selectedUsers.add(randomUser)
                break
    return selectedUsers

# get all unrated movies for a user
def get_unrated_movies(userId):
    rated = df.loc[df['userId'] == userId]
    ratedMovies = set(pd.Series(rated.movieId))
    unratedMovies = allMovieIds.difference(ratedMovies)
    return unratedMovies

# calculate predicted ratings for the yet unrated movies of a user
def get_recommendations(userId, unratedMovies):
    predictions = np.empty((0, 2))
    for movieId in unratedMovies:
        prediction = algo.predict(userId, movieId)
        predictions = np.r_[predictions, [np.array([movieId, prediction.est])]]

    predictionsDf = pd.DataFrame(predictions, columns=['movieIds', 'ratings'])
    predictionsDf.sort_values(by=['ratings'], inplace=True, ascending=False)
    return predictionsDf

# select a movie from the recommendations
def select_movie(userId, predictedRatings):
    randomlySelectedMovie = random.randint(0,4)  # random movie from the top 5 recommendations
    selectedMovieId = predictedRatings.iloc[randomlySelectedMovie]['movieIds']
    return selectedMovieId

# updates the data with a new rating for the selected movie
def update_ratings(userId, movieId):
    newRating = random.randint(1,5)
    df.loc[len(df.index)] = [userId, movieId, newRating]

def evaluate_performance(iteration):
    accuracy = cross_validate(algo, data, cv=5)
    newColumn = 'iteration' + str(iteration)
    performance[newColumn] = np.nan
    performance.at['RMSE', newColumn] = np.mean(accuracy.get('test_rmse'))

def analyse_popularity(iteration):
    #print("analysing popularity" + str(iteration))
    movieRatings = np.empty((0, 3))
    totalNumberOfRatings = 10
    newColumn = 'iteration' + str(iteration)
    globals()['absoluteMovieRatings'][newColumn] = np.nan
    for movieId in allMovieIds:
        absoluteRating = 5 + iteration
        relativeRating = absoluteRating/totalNumberOfRatings
        #movieRatings = np.r_[movieRatings, [np.array([movieId, absoluteRating, relativeRating])]]
        absoluteMovieRatings.at[movieId, newColumn] = absoluteRating
        relativeMovieRatings.at[movieId, newColumn] = relativeRating

    # test whether the data has been updated
    # dataRatingList = data.raw_ratings
    # size = len(dataRatingList)
    # print("updated size: " + str(size))

    #pd.DataFrame(movieRatings, columns=['movieIds', 'absoluteNumberOfRatings', 'relativeNumberOfRatings'])
    #movieRatingsDf.sort_values(by=['absoluteNumberOfRatings'], inplace=True, ascending=False)


# hyperparameter tuning:
# https://medium.com/tiket-com/get-to-know-with-surprise-2281dd227c3e

for i in range(numberOfIterations):
    if i % trainingStep == 0:
        train_algorithm()
    if i % evaluationStep == 0:
        evaluate_performance(i)

    users = select_users()
    for user in users:
        unratedMovies = get_unrated_movies(user)
        recommendations = get_recommendations(user, unratedMovies)
        movie = select_movie(user, recommendations)
        update_ratings(user, movie)

    if i % measuringStep == 0:
        analyse_popularity(i)

print("overall performance:")
print(performance.head())

print("overall ratings:")
print(absoluteMovieRatings.head())
print(relativeMovieRatings.head())