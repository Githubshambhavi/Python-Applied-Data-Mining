
# coding: utf-8

# In[95]:


#Construct a Simulation that demonstrate the flipping a coin 4 times tested 1000 times
import random
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import binom


# In[ ]:


#The simplest way to simulate such a thing is to observe that if x is a
#real number chosen uniformly at random from [0, 1),then P[x<p] = p
# Bernoulli Trial : Only two outcomes :Possiblity of event happening is 1 (p) and not happening is 0 (1-p)

#for Bernoulli Trial possibility of "Heads" is Success that is 1 and Tail is "Failure" i.e 0
def bernoulli(p):
    #Returns a random sample of a Bernoulli random variable with parameter p
    x = random.random() #generate uniform random number X from [0,1], return 1 if x<p
    if x < p:
        return 1 #Return Head
    else:
        return 0 #Return Tail
def binomial(n, p):
    """Returns a random sample of a binomial
    random variable with parameters (n, p)"""
    sum_ans = 0
    for k in range(n):
        sum_ans = sum_ans + bernoulli(p)
    return sum_ans

n = 4
p = 0.5
print ("Random sample of a binomial random variable")
print ("with parameters n =", n, "and p =", p)
print ( binomial(n, p))
print(bernoulli(p))


# In[ ]:


def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def make_hist(n,p,num_points):
    data = [binomial(n,p) for _ in range(num_points)]
    histogram = Counter(data)
    #bar chart to show binomial samples
    plt.bar([x-0.40 for x in histogram.keys()],[v/num_points for v in histogram.values()],0.8,color='0.75')
    mu = p*n
    sigma = math.sqrt(n*p*(1-p))
    
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
    for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()


# In[96]:


make_hist(n,p,10000)

