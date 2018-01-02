
# coding: utf-8

#                   

# In[1]:


# author :  shambhavi srivastava

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random, itertools


# # Vanilla Perceptron Implementation

# In[1057]:


class Perceptron:
    'A simple Perceptron implementation.'
    def __init__(self, weights, bias, alpha=0.1):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
    
    def propagate(self, x):
        return self.activation(self.net(x)) 
        
    def activation(self, net):
        if net > 0:
            return 1
        return 0
    
    def net(self, x):
        return np.dot(self.weights, x) + self.bias
    
    def learn(self, x, y):
        y_hat = self.propagate(x)
        self.weights = [w_i + self.alpha*x_i*(y-y_hat) for (w_i, x_i) in zip(self.weights, x)]
        self.bias = self.bias + self.alpha*(y-y_hat)
        return np.abs(y_hat - y)


# #  1(A) Averaged Perceptron Implementation
# Record the weight vector estimates at each time point and average these to make a final prediction, rather than sticking with only the very last weight vector.
# 
# Below is the Method to calculate the average perceptron. The defination function "Learn" is used to calculate the average preceptron. The idea is to record all the weights and append in the list, after recording all the weight values in the list calculate the average value and update the self.weights.

# In[1058]:


class AveragedPerceptron:
    
    def __init__(self, weights, bias, alpha=0.1):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
    
    def propagate(self, x):
        return self.activation(self.net(x)) 
        
    def activation(self, net):
        if net > 0:
            return 1
        return 0
    
    def net(self, x):
        return np.dot(self.weights, x) + self.bias
    
    def learn(self, x, y):
        y_hat = self.propagate(x)
        list_weight = []
        w_calc = [w_i + self.alpha*x_i*(y-y_hat) for (w_i, x_i) in zip(self.weights, x)]
        list_weight.append(w_calc)
        self.weights = [ sum(all_weights)/len(all_weights) for all_weights in zip(*list_weight)]
        self.bias = self.bias + self.alpha*(y-y_hat)
        return np.abs(y_hat - y)
    


# ## Preparing the data 

# In[1059]:


size = 20
data = pd.DataFrame(columns=('$x_1$', '$x_2$'),
                    data=np.random.uniform(size=(size,2)))
data.head(10)


# # Iterating through the data
# 

# In[1060]:


def learn_data(avg_perceptron,data):
    'Returns the number of errors made.'
    count = 0 
    for i, row in data.iterrows():
        count += avg_perceptron.learn(row[0:2], row[2])
    return count


# In[1061]:


def condition(x):
    return int(np.sum(x) > 1)

data['y'] = data.apply(condition, axis=1)

data.head(10)


# In[1062]:


avg_perceptron = AveragedPerceptron([0.1,-0.1],0.05) 


# # Plotting Data
# 
# The data has been plotted using averaged perceptron calculated.

# In[1063]:


def plot_data(data, ax):
    data[data.y==1].plot(kind='scatter', 
                         x='$x_1$', y='$x_2$', 
                         color='Red', ax=ax)
    data[data.y==0].plot(kind='scatter', 
                         x='$x_1$', y='$x_2$', 
                         color='Gray', ax=ax)
    ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1)
    
def threshold(perceptron, x_1):
    return (-perceptron.weights[0] * x_1 - perceptron.bias) / perceptron.weights[1]

def plot_perceptron_threshold(perceptron, ax):
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    
    x2s = [threshold(perceptron, x1) for x1 in xlim]
    ax.plot(xlim, x2s)
    
    ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1)

def plot_all(perceptron, data, t, ax=None):
    if ax==None:
        fig = plt.figure(figsize=(5,4))
        ax = fig.gca()
    plot_data(data, ax)
    plot_perceptron_threshold(perceptron, ax)
    
    ax.set_title('$t='+str(t+1)+'$')
    


# In[1064]:


f, axarr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8,6))
axs = list(itertools.chain.from_iterable(axarr))
for t in range(6):
    plot_all(avg_perceptron, data, t, ax=axs[t])
    learn_data(avg_perceptron,data)
f.tight_layout()


# # Question 1(B)
#    Weighted Average : To calculate weighted average below formula is used
#    
#    $\frac{(sum of weights)*(t+1)}{21}$
# where    t is time
#          here since the time range from 1 to 6 , calculated the sum of all the time i.e (1+2+..+6)=21 
#          

# In[1065]:


class WeightedAverage:
    
    def __init__(self, weights, bias, alpha=0.1):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
    
    def propagate(self, x):
        return self.activation(self.net(x)) 
        
    def activation(self, net):
        if net > 0:
            return 1
        return 0
    
    def net(self, x):
        return np.dot(self.weights, x) + self.bias
    

    def learn(self, x, y,t):
        y_hat = self.propagate(x)
        # storing the weights as a list.
        weights_values = []
        weights_values.append([w_i + self.alpha*x_i*(y-y_hat) for (w_i, x_i) in zip(self.weights, x)])
        # taking weighted average of weights before updating.
        self.weights = [(sum(col)*(t+1))/21 for col in zip(*weights_values)]
        self.bias = self.bias + self.alpha*(y-y_hat)
        return np.abs(y_hat - y)


# In[1066]:


Wavg_perceptron = WeightedAverage([0.1,-0.1],0.05) 


# In[1067]:


def learn_data_time(perceptron, data,t):
    'Returns the number of errors made.'
    count = 0 
    for i, row in data.iterrows():
        count += perceptron.learn(row[0:2], row[2],t)
    return count


# In[1068]:


f, axarr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16,9))
axs = list(itertools.chain.from_iterable(axarr))
for t in range(6):
    plot_all(Wavg_perceptron, data, t, ax=axs[t])
    learn_data_time(Wavg_perceptron, data,t)
f.tight_layout()


# # 1(C) On sklearn data set

# In[1069]:


from sklearn import datasets

iris = datasets.load_iris()
data2 = pd.DataFrame(columns=('$x_1$', '$x_2$', '$x_3$', '$x_4$'),data=iris.data)

data2['y']=iris.target
data2=data2[:100]
data2.head(10)


# Implementing Vanilla Percptron for iris data set and finding the erros 

# In[1070]:


for t in range(6):
    vanilla_learn = learn_data(avg_perceptron, data2)
    print("Perceptron error value of iris data : " , vanilla_learn)


# Averaged Perceptron for iris dataset and finding errors

# In[1071]:


for t in range(6):
    avg_learn = learn_data(avg_perceptron, data2)
    print("Average Perceptron error value of iris data : " , avg_learn)


# In[1072]:


for t in range(6):
    Wavg_learn = learn_data_time(Wavg_perceptron, data2, t)
    print("Weighted Average Perceptron error value of iris data : " , Wavg_learn)


# # Question 2(A) Naive Bayes hand Calculation

# <table>
# <tr><td>Row #</td><td colspan="6">X</td><td>Y</td></tr>
# <tr><td>1</td><td>C</td><td>S</td><td>S</td><td>S</td><td>T</td><td>S</td><td>Yes</td></tr>
# <tr><td>2</td><td>T</td><td>C</td><td>T</td><td>T</td><td>T</td><td>S</td><td>No</td></tr>
# <tr><td>3</td><td>T</td><td>T</td><td>T</td><td>S</td><td></td><td></td><td>No</td></tr>
# <tr><td>4</td><td>S</td><td>T</td><td>S</td><td>S</td><td>S</td><td></td><td>Yes</td></tr>
# <tr><td>5</td><td>S</td><td>T</td><td>C</td><td>S</td><td>S</td><td></td><td>???</td></tr>
# </table>

# Frequency Table
# 
# <table>
# <tr><td>Row #</td><td>C</td><td>S</td><td>T</td><td>Y</td></tr>
# <tr><td>1</td><td>1</td><td>4</td><td>1</td><td>Yes</td></tr>
# <tr><td>2</td><td>1</td><td>1</td><td>4</td><td>No</td></tr>
# <tr><td>3</td><td>0</td><td>1</td><td>3</td><td>No</td></tr>
# <tr><td>4</td><td>0</td><td>4</td><td>1</td><td>Yes</td></tr>
# <tr><td>5</td><td>1</td><td>3</td><td>1</td><td>???</td></tr>
# </table>
# 
# 

# Using the below Naive Bayes formula, calculate the parameter estimates:
# 
# P(w/c) = $\frac{count(w,c)+1}{count(c)+|V|}$

# P(c/y)=(1+1)/(11+5) = 2/16
# 
# P(s/y)=(8+1)/(11+5) = 9/16
# 
# P(t/y)=(2+1)/(11+5) = 3/16
# 
# 
# P(c/y')=(1+1)/(10+5) = 2/15
# 
# P(s/y')=(2+1)/(10+5) = 3/15
# 
# P(t/y')=(7+1)/(10+5) = 8/15
# 
# 
# 
# P(y) = 2/4
# 
# P(y') = 2/4

# # 2(B) Prediction

# P(y/x5)= 2/4 * 2/16 * (9/16)^3 * 3/16 = 0.002085
# 
# 
# P(y'/x5)= 2/4 * 2/15 * (3/15)^3 * 8/15 = 0.000284444
# 

# # Question 3 
# Logistic Regression in sklearn

# In[1073]:


import sklearn
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
from sklearn.svm import SVC
from sklearn import cross_validation

from sklearn import datasets
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# In[1074]:


movie_data = load_files("movie-reviews/")
print("n_samples: %d" % len(movie_data.data))


# In[1075]:


docs_train, docs_test, y_train, y_test = train_test_split(
        movie_data.data, movie_data.target, test_size=0.25, random_state=None)


# In[1076]:


vectorizer = CountVectorizer(stop_words="english")
# fit the vectorizer to the training documents
X_train = vectorizer.fit_transform(docs_train)


# In[1077]:


model = MultinomialNB()
model.fit(X_train, y_train)
X_test = vectorizer.transform(docs_test)
y_hat = model.predict(X_test)

print("accuracy is: ", metrics.accuracy_score(y_test, y_hat))

print("classification report: ", metrics.classification_report(y_test, y_hat))


# ### 3 (A) Logistic Regression

# In[1078]:


lr = linear_model.LogisticRegression()
lr.fit(X_train, y_train)


# In[1079]:


y_hat2 = lr.predict(X_test)


# In[1080]:


print("accuracy is: ", metrics.accuracy_score(y_test, y_hat2))


# In[1081]:


print("classification report: ", metrics.classification_report(y_test, y_hat2))


# 3 (B) LR using Binary Encoding

# In[1082]:


vectorizer2 = CountVectorizer(binary=True,stop_words="english")

# fit the vectorizer to the training documents
X_train3 = vectorizer2.fit_transform(docs_train)


# In[1083]:


lr_binary = linear_model.LogisticRegression()
lr_binary.fit(X_train3, y_train)


# In[1084]:


X_test3 = vectorizer2.transform(docs_test)
y_hat3 = lr_binary.predict(X_test)


# In[1085]:


print("accuracy is: ", metrics.accuracy_score(y_test, y_hat3))


# When calculated the accuary metrics for Logistic regression with and without binary code , the result is the accuray is more defined while using "Binary Encoding" 

# # 3(C) Performing Cross-fold validation.

# In[1086]:


iris = datasets.load_iris()

print(iris.data.shape)


# In[1087]:


X = iris.data[:1000]
y = iris.target[:1000]


# In[1088]:


kf_total = cross_validation.KFold(len(X), n_folds=10, shuffle=True, random_state=4)
for train, test in kf_total:
    print (train, '\n', test, '\n\n')


# Logistic Regression using ten-fold cross-validation

# In[1089]:


lr_cv = linear_model.LogisticRegression()
lr_score = [lr_cv.fit(X[train_indices], y[train_indices]).score(X[test_indices],y[test_indices])
for train_indices, test_indices in kf_total]

print(lr_score)


# In[1090]:


nb_cv = MultinomialNB()
nb_score = [nb_cv.fit(X[train_indices], y[train_indices]).score(X[test_indices],y[test_indices])
for train_indices, test_indices in kf_total]

print(nb_score)


# BOX PLOT COMPARISION

# In[1091]:


data_to_plot = [lr_score , nb_score]

fig = plt.figure(1, figsize=(10, 8))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

# Save the figure
fig.savefig('fig1.png', bbox_inches='tight')

plt.xticks([1, 2], ['LR', 'NB'])

