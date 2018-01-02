
# coding: utf-8

# #####         Shambhavi Srivastava

# In[294]:


import pandas as pd
import numpy as np

import sklearn.metrics as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')


# #### Introduction
# 
# The project aims to utilize the music rating database of Yahoo! Music to develop a rating predictor system. The database contains list of users (identifiable by User ID) who have provided Rating for a particular song (identifiable by Song ID).<b>Using these three metrics, a rating predictor system is devised which could predict a particular user’s rating for a new song </b>. 

# In[295]:


# Read the dataset
yahoo_data = pd.read_csv("ydata-ymusic-rating-study-train.txt",header=None,delimiter='\t', quoting=3 ,
                               names= ['userId','SongId','Ratings'])

yahoo_data.shape


# In[296]:


yahoo_data.head()


#  ### Qs 1(A) : Exploratory statistics and plots

# ### Rating-Frequency Plot 
# 
# Below table shows the rating and its frequency :
# 
# | 1 	| 97844 	|
# |---	|-------	|
# | 2 	| 39652 	|
# | 3 	| 49131 	|
# | 4 	| 48480 	|
# | 5 	| 76597 	|

# In[297]:


# Plot rating-frequency 
fig, ax = plt.subplots()
yahoo_data['Ratings'].value_counts(sort=False).plot(ax=ax, kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Rating')


# #### Summary Stats : User based 

# In[298]:


# Dataframe with relation between user and ratings, drop the song id column
yahoo_user_data = yahoo_data.drop(["SongId"], axis=1)
yahoo_user_data.head()

# Calculate Average rating given by each user, and its SD.
yahoo_user_ratings= yahoo_user_data.groupby(["userId"]).mean()
yahoo_user_ratings.columns = ["Mean Rating"]
yahoo_user_ratings["Standard Deviation"] = yahoo_user_data.groupby(["userId"]).std()
yahoo_user_ratings["Count"] = yahoo_user_data.groupby(["userId"]).transform("count")
yahoo_user_ratings.head()

fig, ax = plt.subplots()
yahoo_user_ratings['Mean Rating'].plot(ax=ax, kind='hist')
plt.ylabel('Frequency')
plt.xlabel('Mean Users Rating')
yahoo_user_ratings.head()


# #### Summary Stats : Song based 

# In[299]:


# Data define the relation between songs and its ratings
yahoo_song_data = yahoo_data.drop(["userId"], axis=1)
yahoo_song_data.head()

#Calculate the Average rating given to each song and its SD
yahoo_song_ratings = yahoo_song_data.groupby(["SongId"]).mean()
yahoo_song_ratings.columns = ["Mean Rating"]
yahoo_song_ratings["Standard Deviation"] = yahoo_song_data.groupby(["SongId"]).std()
yahoo_song_ratings["Count"] = yahoo_song_data.groupby(["SongId"]).transform("count")


yahoo_song_small = yahoo_song_ratings[:1000]
fig, ax = plt.subplots()
yahoo_song_ratings['Mean Rating'].plot(ax=ax, kind='hist', color='k')
plt.ylabel('Frequency')
plt.xlabel('Mean Songs Rating')
yahoo_song_ratings.head()


# <i> For above summary stats : song based not surprisingly, most songs have intermediate mean rating and only few items have a mean rating on extreme high or low ends of the scale. Indeed, the distribution of song mean ratings follow a unimodal distribution, with a mode at 2.8 as shown above.</i>

# <i>Interestingly, the distribution of song mean ratings presented in above is slightly different from the distribution of mean ratings of users, depicted before: the distribution is now bit skewed, with mode shifted to a mean rating of 3.8(approx). Different rating behavior of users accounts for the apparent difference between the distributions. It turns out that users who rate more songs tend to have considerably lower mean ratings.</i>

# ### Qs 1(B) : Clustering

# ###### Features Selection : 
# 
# <i>Below are expanded feature selection for the process of clustering. After analyses the below features seems to have higher correlations to the ratings</i>
# 
# ##### User -based statistical features
# 1. User's average rating on song x
# 2. User's rating count on song x 
# ##### Songs-based statistical features
# 4. Number of ratings 
# 5. Song's rating mean

# #### Data Processing: 
# 
# <i>For better evaluation split the data set into two 80% and 20% for training and testing</i>

# In[300]:


#Split test and train  data
yahoo_train, yahoo_test = train_test_split(yahoo_data, test_size = 0.2)
print(len(yahoo_train))
yahoo_train.head()


# In[301]:


# split training data into data and target
yahoo_data = yahoo_train.drop(yahoo_train[[2]],axis=1)
yahoo_target= yahoo_train.drop(yahoo_train[[0,1]],axis=1)


# ###### Below block redfine the feature as User-based Features i.e User's average rating on song x , User's rating count on song x

# In[302]:


# User-based Features
# User's average rating on song x
user_feat = yahoo_train.drop("userId",1)
user_avg_rating = user_feat[['SongId', 'Ratings']].groupby(["SongId"], as_index = False).mean()
# User's rating count on song x
user_ctn_ratin = user_feat[['SongId', 'Ratings']].groupby(["SongId"], as_index = False).count()


# ###### Merge the above two Data frames for proper clustering  and then split the data into train and test set

# In[303]:


#merge two data-sets on songid 
df_new = pd.merge(user_avg_rating,user_ctn_ratin, on='SongId')
df_new.head()

#Split into test and train

df_new_train,df_new_test = train_test_split(df_new, test_size=0.25)
df_new_train.head()


# ##### K-Means Clustering 
# <i>This learning model involves both the input(x) and output(y). During the learning process the error between the predicted outcome and the actual outcome is used to train the system. 
# 
# Below block of code train the model for one particular user and predict the rating by user based on rating given by user in past. For this K-means is used to separate the data into <b>three clusters</b> using features as <b>"Rating Mean" and "Count of Rating"</b> w.r.t the user.The target is the actual rating provided by the user.</i>

# In[304]:


yahoo_user_11= yahoo_train.loc[yahoo_train["userId"]==1]
yahoo_user_11.head()

# merge dataframe
df_main = pd.merge(yahoo_user_11,df_new_train, on='SongId')
print(df_main)
len_rows=len(df_main)


# In[305]:


# K-Means 
feat_cols = ["Ratings_x" , "Ratings_y"]
yahoo_data = df_main[feat_cols] #data
x = pd.DataFrame(yahoo_data)
x.columns = ["Rating_mean" , "Count"]

yahoo_target = df_main.Ratings #target
y = pd.DataFrame(yahoo_target)
y.columns = ["Targets"]

np.unique(y)

x.head()


# ##### Visualise the data
# <i>For data visualisation scatter plot is used looking at the Rating mean and number of ratings assigned to the song </i>

# In[306]:


# Set plot
plt.figure()
# Plot features (Rating mean , Count of Ratings)
plt.subplot()
plt.scatter(x.Rating_mean, x.Count, c=y.Targets)
plt.title('Data Visualisation Intial')


# #### Build the K-means Model :
# 
# <i>For this first create the model and specify the number of clusters the model should find(n_clusters=3) and then fit the model</i>

# In[307]:


from sklearn.cluster import KMeans
# K Means Cluster
model = KMeans(n_clusters=3)
model.fit(x)

# view the results 
model.labels_


# ##### Visualise the classifer results : 
# <i>Plot the actual classes against the predicted classes from the K Means model</i>

# In[308]:


# Set the size of the plot
plt.figure(figsize=(14,7))
 
# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Rating_mean, x.Count, c=y.Targets, s=40)
plt.title('Real Classification')
 
# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(x.Rating_mean, x.Count, c=model.labels_, s=40)
plt.title('K Mean Classification')


# ##### Performance Measures : 
# <i>In order to measure the performance calculated the accuracy and also the confusion matrix.
# 
# For this we need two values of y(target) which is the true(original) values and predicted models values.</i>

# In[309]:


# Performance Metrics
sm.accuracy_score(y, model.labels_)


# In[310]:


# Confusion Matrix
sm.confusion_matrix(y, model.labels_)


# ### Qs 2(A) Story-telling and/or hypothesis generation

# #### Methodology:
# <i>The metric Song ID in itself is not a meaningful number for analysis since it merely represents a song. However, Song ID can be utilize to process the dataset and develop a set of features which could be used for KNN algorithm. 
# 
# For a particular user, the rating predictor mechanism is implemented as follows:
# 
# <b>Step1.</b> For a given User ID, find the songs rated by the user and the rating provided by him/her. Lets call the list of ratings as ‘user_rating_target’
# 
# <b>Step2.</b> For each song found in Step1, find the number of ratings assigned to the song. Lets call this data set as ‘Ratings_y’
# 
# <b>Step3.</b> For each song found in Step1, find the mean of the ratings (calculated over the complete dataset). Lets call this mean rating dataset as ‘Ratings_x’
# 
# <b>Step4.</b>. Using “Ratings_x” and “Ratings_y” as the predicting features and by using ‘user_rating_target’ as the target, KNN algorithm is implemented. The underlying idea is to use the number of ratings for a given song and its mean rating to predict the rating given by a particular user.
# 
# For any number of dataset entries available for a given User ID, 75% data is used for training the model and 25% is used for testing the model.</i>
# 

# #### K-NN (K-Nearest Neighbors)

# In[311]:


# K-NN
# user with ID 1
yahoo_user_11= yahoo_train.loc[yahoo_train["userId"]==1]
yahoo_user_11.head()

df_main = pd.merge(yahoo_user_11,df_new_train, on='SongId')
print(df_main)
len_rows=len(df_main)


# In[312]:


x= df_main["Ratings_x"]
y = df_main["Ratings_y"]
plt.scatter(x,y,color="purple")


# In[313]:


# KNN
feat_cols = ["Ratings_x" , "Ratings_y"]
yahoo_data = df_main[feat_cols] #data
yahoo_target = df_main.Ratings #target

np.unique(yahoo_target)


# In[314]:


yahoo_X_train , yahoo_X_test = train_test_split(yahoo_data, test_size=0.25)
yahoo_y_train, yahoo_y_test = train_test_split(yahoo_target, test_size=0.25)

yahoo_X_test.shape


# In[315]:


# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(yahoo_X_train, yahoo_y_train) 
y_hat = knn.predict(yahoo_X_test)
print(metrics.accuracy_score(yahoo_y_test, y_hat))


# ##### Plot the  accuarcies score for train and test data

# In[316]:


train_accuracies, test_accuracies = [], []
k_vals = list(range(1,int(len_rows*0.7)))
print(k_vals)
for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(yahoo_X_train, yahoo_y_train)
    y_train_hat = knn.predict(yahoo_X_train)
    y_test_hat  = knn.predict(yahoo_X_test)
    test_accuracies.append(metrics.accuracy_score(yahoo_y_test, y_test_hat))
    train_accuracies.append(metrics.accuracy_score(yahoo_y_train, y_train_hat))

plt.plot(k_vals, train_accuracies, label='train perf')
plt.plot(k_vals, test_accuracies, label='test perf')
plt.legend();


# <i> Consider the data graph for User ID 1. By plotting the accuracy(fig above) as a function of the ‘number of nearest neighbours’ (k), it can be seen that for test data the accuracy improves as compared to the training data. Further, it seems that for k>7 or higher, the model tends be having a consistent low inaccuracy.</i>
