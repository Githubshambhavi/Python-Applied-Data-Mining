
# coding: utf-8

# In[125]:


import numpy as np 
import pandas as pd

import sklearn 
from sklearn.cluster import KMeans
from sklearn import datasets

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

import math


# In[126]:


dataset = datasets.load_boston()


# In[127]:


df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
#Shape of the dataset each column is an attribute ('feature')
print(dataset.feature_names) #print columns of the dataset

#Print first 10 rows
print(dataset.data[:10])


# In[128]:


df.head()


# In[129]:


crime = df["CRIM"].values
zn = df["ZN"].values
indus = df["INDUS"].values
chas = df["CHAS"].values
nox = df["NOX"].values
rm = df["RM"].values
age = df["AGE"].values
dis = df["DIS"].values
rad = df["RAD"].values
tax = df["TAX"].values
ptratio = df["PTRATIO"].values
b = df["B"].values
lstat = df["LSTAT"].values


# In[130]:


# Calculate mean for each column

def estimate_mean(data):
    return sum(data)/len(data)
print("mean of crime col is :" ,estimate_mean(crime))
print("mean of ZN col is :" ,estimate_mean(zn))
print("mean of INDUS col is :" ,estimate_mean(indus))
print("mean of CHAS col is :" ,estimate_mean(chas))
print("mean of NOX col is :" ,estimate_mean(nox))
print("mean of RM col is :" ,estimate_mean(rm))
print("mean of AGE col is :" ,estimate_mean(age))
print("mean of DIS col is :" ,estimate_mean(dis))
print("mean of RAD col is :" ,estimate_mean(rad))
print("mean of TAX col is :" ,estimate_mean(tax))
print("mean of PTRATIO col is :" ,estimate_mean(ptratio))
print("mean of B col is :" ,estimate_mean(b))
print("mean of LSTAT col is :" ,estimate_mean(lstat))


# In[131]:


# Calculate Variance for all columns

def estimate_variance(data, mu=None):
    if mu is None:
        mu = estimate_mean(data)
    return sum([(x - mu)**2 for x in data]) / len(data)

print("Variance of CRIME col is :" ,estimate_variance(crime))
print("Variance of ZN col is :" ,estimate_variance(zn))
print("Variance of INDUS col is :" ,estimate_variance(indus))
print("Variance of CHAS col is :" ,estimate_variance(chas))
print("Variance of NOX col is :" ,estimate_variance(nox))
print("Variance of RM col is :" ,estimate_variance(rm))
print("Variance of AGE col is :" ,estimate_variance(age))
print("Variance of DIS col is :" ,estimate_variance(dis))
print("Variance of RAD col is :" ,estimate_variance(rad))
print("Variance of TAX col is :" ,estimate_variance(tax))
print("Variance of PTRATIO col is :" ,estimate_variance(ptratio))
print("Variance of B col is :" ,estimate_variance(b))
print("Variance of LSTAT col is :" ,estimate_variance(lstat))


# In[132]:


#1(c) Scatter 'NOX' vs 'CRIM' 

plt.scatter(nox, crime)
plt.xlabel("Nitric Oxide Concentration")
plt.ylabel("Per Captia Crime")


# In[133]:


#1(c) Scatter "CRIME" vs "The Housing Prices"

df["price"]= dataset.target
housing_price = df["price"]
plt.scatter(crime, housing_price)
plt.xlabel("Crime")
plt.ylabel("Housing Prices")



# In[136]:


#1(d) Correlations between two pairs "NOX' and "CRIM"
 #Correlation lies between -1(perfect anti correlation) to 1(perfect correlation) # number like 0.25 is a weak positive correlation
   
 

def std_dev(data):  #Calculate standard deviation
    return math.sqrt(estimate_variance(data))

def covariance(x,y): #Covariance:how two variables vary in tandem from their means
    n = len(x)
    return np.dot(estimate_mean(x),estimate_mean(y))/(n-1)

def correlation(x,y): #divide out standard deviations of both the variables.
    stddev_x = std_dev(x)
    stddev_y = std_dev(y)
    if stddev_x > 0 and stddev_y > 0 :
        return covariance(x,y)/ stddev_x/ stddev_y
    else:
        return 0 #if no variation then correlation is 0
    
print("Correlation between NOX and Crime is : " ,correlation(nox,crime))
print("Correlation between Crime and Housing Prices is : " ,correlation(crime,housing_price))


