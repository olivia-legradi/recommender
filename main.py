import numpy as np
from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate, GridSearchCV
import pandas as pd
import random

from libreco.data import random_split, DatasetPure
from libreco.algorithms import LightGCN  # pure data, algorithm LightGCN
from libreco.evaluation import evaluate


#variables for the movielens dataset:
#df = pd.read_csv("ml-latest-small/ratings.csv")
#df.drop("timestamp", inplace=True, axis=1)
#moviesDf = pd.read_csv("ml-latest-small/movies.csv")
#allMovieIds = set(pd.Series(moviesDf.movieId).unique())
#absoluteMovieRatings = pd.DataFrame(index=pd.Index(allMovieIds))
#relativeMovieRatings = pd.DataFrame(index=pd.Index(allMovieIds))
#reader = Reader(rating_scale=(1,5))

#variables for the LFM-360k dataset
#columns = ['userId', 'artistId', 'artistName', 'rating']
#df = pd.read_table("lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv", header=None, names=columns)
#df.drop('artistName', inplace=True, axis=1)
#artistColumns = ['gender']
#artistsDict = pd.read_json('lfm-360-gender/lfm-360-gender.json', typ='dictionary')
#artistsDf = pd.DataFrame(artistsDict, columns=artistColumns)


#reader = Reader(rating_scale=(1,419157))
#data = Dataset.load_from_df(df, reader)

#performance = pd.DataFrame(index=pd.Index(['RMSE']))
#algo = SVD()

numberOfIterations = 2
numberOfUsers = 2
trainingStep = 2
evaluationStep = 2
measuringStep = 2

#cleans the data
def clean_data():
    columns = ['userId', 'artistId', 'artistName', 'rating']
    df = pd.read_table("lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv", header=None, names=columns)
    df.drop('artistName', inplace=True, axis=1)

    #filter out artistIDs where gender is None
    artistColumns = ['gender']
    artistsDict = pd.read_json('lfm-360-gender/lfm-360-gender.json', typ='dictionary')
    artistsDf = pd.DataFrame(artistsDict, columns=artistColumns)

    artistsWithGender = artistsDf[~artistsDf['gender'].isnull()]
    artistIdswithGender = list(artistsWithGender.index.values)
    artistIdswithGender = list(dict.fromkeys(artistIdswithGender))
    ratingsWithGender = df[df['artistId'].isin(artistIdswithGender)]

    #filter out rows with a rating higher than a limit
    ratingsWithGender = ratingsWithGender.astype({'rating': 'int'})
    ratingLimit = 2000
    cleanRatings = ratingsWithGender[ratingsWithGender['rating'] <= ratingLimit]

    cleanRatings.to_csv('cleanRatings.csv', index=False)

    #create a testset with n users
    numberOfUsers = 1000
    selectedUsers = set()
    allUsers = np.array(pd.Series(cleanRatings.userId).unique())
    print('# of all users: ')
    print(len(allUsers))
    for i in range(numberOfUsers):
        while True:
            randomNumber = random.randint(0, len(allUsers) - 1)
            randomUser = allUsers[randomNumber]
            if not randomUser in selectedUsers:
                selectedUsers.add(randomUser)
                break

    testSet = cleanRatings[cleanRatings['userId'].isin(selectedUsers)]
    print('size of testset:')
    print(len(testSet))
    testSet.to_csv('testSet.csv', index=False)

# tunes the hyperparameters
# def tune_hyperparameters():
#     param_grid = {
#         'n_factors': [20, 50, 100],
#         'n_epochs': [5, 10, 20]
#     }
#     gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
#     gs.fit(data)
#     bestFactor = gs.best_params['rmse']['n_factors']
#     bestEpoch = gs.best_params['rmse']['n_epochs']
#     globals()['algo'] = SVD(n_factors=bestFactor, n_epochs=bestEpoch)

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

# calculates the RSME -> VERALTET, Surprise package
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

#clean_data()

#retrains the model after n-number of iterations
def retrain_model():
    train_data, new_data_info = DatasetPure.merge_trainset(train_data, loaded_data_info, merge_behavior=True) --> but DatasetPure
    #eval_data = DatasetFeat.merge_evalset(eval_data, new_data_info)  # use new_data_info

data = pd.read_csv('testSet.csv', names=['user', 'item', 'label'], skiprows=1)

print('data: ')
print(data.head())

# split whole data into three folds for training, evaluating and testing
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)
print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

lightgcn = LightGCN(
    task="ranking",
    data_info=data_info,
    loss_type="bpr",
    embed_size=16,
    n_epochs=3,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
    device="cuda",
)
# monitor metrics on eval data during training
lightgcn.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    eval_data=eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)

# do final evaluation on test data
evaluate(
    model=lightgcn,
    data=test_data,
    neg_sampling=True,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)

print('making prediction:')
# predict preference of user 00087f98b9db64d761d3f1404e1558da9245137a to item 05517043-ff78-4988-9c22-88c68588ebb9
score = lightgcn.predict(user='00087f98b9db64d761d3f1404e1558da9245137a', item='05517043-ff78-4988-9c22-88c68588ebb9')
print(score)

print('making recommendation')
# recommend 7 items for user 00087f98b9db64d761d3f1404e1558da9245137a
recommendation = lightgcn.recommend_user(user='00087f98b9db64d761d3f1404e1558da9245137a', n_rec=7)
print(recommendation)

#
# for i in range(numberOfIterations):
#     if (i % trainingStep == 0):
#         train_algorithm()
#     #if i % evaluationStep == 0:
#         #evaluate_performance(i)
#
#     users = select_users()
#     for user in users:
#         #unratedMovies = get_unrated_movies(user)
#         #recommendations = get_recommendations(user, unratedMovies)
#         #movie = select_movie(user, recommendations)
#         #selectedItem = select_item(user, recommendations)
#         #update_ratings(user, movie)
#         possibleItems = np.array(pd.Series(df.artistId).unique())
#         recommendations = get_recommendations(user, possibleItems)
#         analyse_gender(recommendations)
#
#     #if i % measuringStep == 0:
#         #analyse_popularity(i)
