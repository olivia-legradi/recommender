import numpy as np
from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate, GridSearchCV
import pandas as pd
import random

#variables for the movielens dataset:
#df = pd.read_csv("ml-latest-small/ratings.csv")
#df.drop("timestamp", inplace=True, axis=1)
#moviesDf = pd.read_csv("ml-latest-small/movies.csv")
#allMovieIds = set(pd.Series(moviesDf.movieId).unique())
#absoluteMovieRatings = pd.DataFrame(index=pd.Index(allMovieIds))
#relativeMovieRatings = pd.DataFrame(index=pd.Index(allMovieIds))
#reader = Reader(rating_scale=(1,5))

#variables for the LFM-360k dataset
columns = ['userId', 'artistId', 'artistName', 'rating']
df = pd.read_table("lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv", header=None, names=columns)
df.drop('artistName', inplace=True, axis=1)
artistColumns = ['gender']
artistsDict = pd.read_json('lfm-360-gender/lfm-360-gender.json', typ='dictionary')
artistsDf = pd.DataFrame(artistsDict, columns=artistColumns)


reader = Reader(rating_scale=(1,419157))
data = Dataset.load_from_df(df, reader)

performance = pd.DataFrame(index=pd.Index(['RMSE']))
algo = SVD()

numberOfIterations = 2
numberOfUsers = 2
trainingStep = 2
evaluationStep = 2
measuringStep = 2

# tunes the hyperparameters
def tune_hyperparameters():
    param_grid = {
        'n_factors': [20, 50, 100],
        'n_epochs': [5, 10, 20]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
    gs.fit(data)
    bestFactor = gs.best_params['rmse']['n_factors']
    bestEpoch = gs.best_params['rmse']['n_epochs']
    globals()['algo'] = SVD(n_factors=bestFactor, n_epochs=bestEpoch)

# trains the algorithm
def train_algorithm():
    print('start training')
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    print('training completed')

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

# gets all unrated movies for a user
def get_unrated_movies(userId):
    rated = df.loc[df['userId'] == userId]
    ratedMovies = set(pd.Series(rated.movieId))
    unratedMovies = allMovieIds.difference(ratedMovies)
    return unratedMovies

# calculates predicted ratings for the possibilities
def get_recommendations(userId, possibleItems):
    predictions = np.empty((0, 2))
    for itemId in possibleItems:
        print('getting predicted rating for item ')
        print(itemId)
        prediction = algo.predict(userId, itemId)
        print('predicted rating is ')
        print(prediction)
        predictions = np.r_[predictions, [np.array([itemId, prediction.est])]]

    predictionsDf = pd.DataFrame(predictions, columns=['itemIds', 'ratings'])
    predictionsDf.sort_values(by=['ratings'], inplace=True, ascending=False)
    return predictionsDf

# selects a movie from the recommendations
# def select_movie(userId, predictedRatings):
#     randomlySelectedMovie = random.randint(0,4)  # random movie from the top 5 recommendations
#     selectedMovieId = predictedRatings.iloc[randomlySelectedMovie]['movieIds']
#     return selectedMovieId

# updates the data with a new rating for the selected movie
# def update_movie_ratings(userId, movieId):
#     newRating = random.randint(1,5)
#     df.loc[len(df.index)] = [userId, movieId, newRating]

# calculates the RSME
def evaluate_performance(iteration):
    accuracy = cross_validate(algo, data, cv=5)
    newColumn = 'iteration' + str(iteration)
    performance[newColumn] = np.nan
    performance.at['RMSE', newColumn] = np.mean(accuracy.get('test_rmse'))

# calculates the absolute and relative popularity of every movie
# def analyse_popularity(iteration):
#     totalNumberOfRatings = len(df)
#     newColumn = 'iteration' + str(iteration)
#     globals()['absoluteMovieRatings'][newColumn] = np.nan
#     for movieId in allMovieIds:
#         ratingsPerMovie = df.loc[df['movieId']==movieId]
#         absoluteRating = len(ratingsPerMovie)
#         relativeRating = absoluteRating/totalNumberOfRatings
#         absoluteMovieRatings.at[movieId, newColumn] = absoluteRating
#        relativeMovieRatings.at[movieId, newColumn] = relativeRating

def analyse_gender(recommendations):
    firstFemale = None
    firstMale = None
    numberOfFemales = 0
    numberOfMales = 0

    for ind in recommendations.index:
        artistId = recommendations['itemIds'][ind]
        gender = artistsDf.at[artistId, 'gender']
        if gender == 'Female':
            if firstFemale is None:
                firstFemale = ind
            numberOfFemales = numberOfFemales + 1
        elif gender == 'Male':
            if firstMale is None:
                firstMale = ind
            numberOfMales = numberOfMales + 1



#tune_hyperparameters()

for i in range(numberOfIterations):
    if (i % trainingStep == 0):
        train_algorithm()
    #if i % evaluationStep == 0:
        #evaluate_performance(i)

    users = select_users()
    for user in users:
        #unratedMovies = get_unrated_movies(user)
        #recommendations = get_recommendations(user, unratedMovies)
        #movie = select_movie(user, recommendations)
        #selectedItem = select_item(user, recommendations)
        #update_ratings(user, movie)
        possibleItems = np.array(pd.Series(df.artistId).unique())
        recommendations = get_recommendations(user, possibleItems)
        analyse_gender(recommendations)

    #if i % measuringStep == 0:
        #analyse_popularity(i)
