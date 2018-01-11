import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import math, random
from collections import defaultdict, Counter


names=['user_id','item_id','rating','timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)

n_users = df.user_id.unique().shape[0]
print("n users: %s" % n_users)
n_items = df.item_id.unique().shape[0]
print("n items: %s" % n_items)

#user-item matrix

ratings = np.zeros((n_users,n_items))
print(ratings)

for row in df.itertuples():
    #row[1] will be user_id and row[2] will be rating
    ratings[row[1]-1, row[2]-1] = row[3]
print(ratings)

#how many zero's are there in the matrix (sparsity in matrix)
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print('sparsity: {:4.2f}%'.format(sparsity))

#function split train and test data
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        ###
        # sample 10 ratings from each user to use
        # as 'test' data
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        # effectively remove these from the training
        # set
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # make sure that test and training are truly disjoint
    assert(np.all((train * test) == 0))

    return train, test

train,test=train_test_split(ratings)
print(train)

#similarity 
def similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
        print(sim)
    elif kind == 'item':
        # we need only flip the dimensions around
        # to do item based similarity!
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / (norms * norms.T))

def predict(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

user_similarity = similarity(ratings,kind='user')
item_similarity = similarity(ratings, kind='item')
item_prediction = predict(train, item_similarity, kind='item')
print(item_prediction)
user_prediction = predict(train, user_similarity, kind='user')
print(user_prediction)
print('User-based CF MSE: ' + str(get_mse(user_prediction, test)))
print('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))
