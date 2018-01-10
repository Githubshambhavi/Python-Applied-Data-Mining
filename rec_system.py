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


# example from data science from scratch

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

unique_interest = sorted(list({interest
                              for user_interest in users_interests
                              for interest in user_interest}))


#create a user_interest vector
def user_interest_vector(user_interests):
    return[1 if interest in user_interests else 0
          for interest in unique_interest]

user_interest_matrix = list(map(user_interest_vector,users_interests))


def cosine_similarity(v, w):
    return np.dot(v, w) / math.sqrt(np.dot(v, v) * np.dot(w, w))

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                     for interest_vector_i in user_interest_matrix]



def most_similar_users(user_id):
    pairs = [(other_user_id, similarity)                      
             for other_user_id, similarity in                 
                enumerate(user_similarities[user_id])         
             if user_id != other_user_id and similarity > 0]  

    return sorted(pairs,                                      
                  key=lambda pair: pair[1],
                  reverse=True)


def user_based_suggestions(user_id, include_current_interests=False):
    # sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
            
# convert them to a sorted list
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[1],
                         reverse=True)

    # and (maybe) exclude already-interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

