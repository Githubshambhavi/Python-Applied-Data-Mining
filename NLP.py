
# coding: utf-8

# #####  Shambhavi Srivastava

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import glob
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import numpy as np

import string

from gensim import corpora, models


# The below code merge together the two text files and write it in one single text file named "merged-debate.txt".

# In[2]:


read_files = glob.glob("*.txt")
with open("merged-debate.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
         


# In[3]:


debate_data = pd.read_csv("merged-debate.txt",header=None,delimiter='\n', quoting=3, names= ["sentence"])
debate_data.shape


# #### Total Number of Instances : 3081

# In[4]:


debate_data.head()


# Below defination is used to clean the data, method incorporates below functions:
# 1. Remove the HTML tags if any using Beautiful soap package.
# 2. Remove the non-sense words using regular expression: like " 's,'re,'n't"
# 3. Tokenizing, i.e converting document to its atomic elements
# 4. Stop words for words like " for,or,and"
# 

# In[5]:


from bs4 import BeautifulSoup
import re
debate_doc = debate_data["sentence"].tolist()

def tokenzied_data(raw_data):

    tokenized_docs = []
    for doc in raw_data:
    # lowercase the words
        doc_lowercase = doc.lower()
    #remove HTML
        doc_text = BeautifulSoup(doc_lowercase).get_text() 
    #Remove non-sense words like ('s,'re,n't)   
        doc_letters_only = re.sub("[^a-zA-Z]", " ", doc_lowercase)
    # now tokenize via NLTK
        doc_tokens = nltk.tokenize.word_tokenize(doc_letters_only)
    # drop stop words, like 'the', 'a', etc.
        stop_list = stopwords.words('english')
        stop_list.extend(string.punctuation)
    #Remove common words and tokenize
        doc_tokens = [word for word in doc_tokens if not word in stop_list]
         
        tokenized_docs.append(doc_tokens)
    return tokenized_docs


# In[6]:


debate_text = tokenzied_data(debate_doc)

print(debate_text)


# Stemming :  Reduce topically similar words to their root. For example, “stemming,” “stemmer,” “stemmed,” all have similar meanings; stemming reduces those terms to “stem.”

# In[7]:


p_stemmer = PorterStemmer()
tokenized_and_stemmed = [[p_stemmer.stem(w) for w in doc] for 
                             doc in debate_text]
print(tokenized_and_stemmed[0])


# #### How many words are there total? 4156 unique words

# In[51]:


dictionary = corpora.Dictionary(tokenized_and_stemmed)
print(dictionary)


# In[9]:


print(dictionary.token2id)


# In[10]:


corpus = [dictionary.doc2bow(text) for text in tokenized_and_stemmed]
print(corpus)


# # 1(B) LDA Implementation

# In[11]:


#With K=5
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, 
                                            id2word = dictionary, 
                                            passes=20)


# In[12]:


lda_model.print_topics(num_topics=5,num_words=3)


# In[13]:


#With K=10
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, 
                                            id2word = dictionary, 
                                            passes=20)
lda_model.print_topics(num_topics=10,num_words=3)


# In[14]:


#With K=20
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, 
                                            id2word = dictionary, 
                                            passes=20)
lda_model.print_topics(num_topics=20,num_words=3)


# # Convert the topics into just a list of the top 20 words in each topic

# In[15]:


topics_matrix = lda_model.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix)

topic_words = topics_matrix[:,:,1]
for i in topic_words:
    print([str(word) for word in i])
    print()


# # Which k do you think is best for this data and why?
# 
# ### When the document is divided into number of topics as "5".
# 
# For above dataset , each topic is separated by comma. Within each topic there are three most probable words to appear in the topic. The document set is not so big. Some things to analyse will be like-- the third topic "need + isi + world'  make sense together. The first topic seems bit confusing 'peopl + 0.015*go + 0.013*countri'.
# 
# #### Although using larger value of k it is observed that the results tend to provide better/accurate results. However, a larger dataset along with relevant contents would also be a determining factor that how effective the larger k value can be for example if the dataset is quite small and comparable to the value of K the model would be not that accurate or effective : 
# Based on the results I got on the experiment I can clearly see that by using K=20, (and number of words=3) combination of words in each topic tend to signify more solidtiy that the topics are related to a political debate/discussion. with K=5 nearly all combination within each topics doesn't signifying anything related to a presidential debate example like " '0.017*tax + 0.015*peopl + 0.015*go','0.028*peopl + 0.018*go + 0.014*care','0.031*applaus + 0.025*mdash + 0.020*let' ".
# 
# ###### Using LDA method would produce some topics that make no sense. So it depends on the context if we want to interpret the topics or just get a set of features

# # 1(c) Topic Modelling separately for files :
# 

# In[16]:


debate_democrat = pd.read_csv("merged-democratic-debate-transcripts.txt",header=None,delimiter='\n',
                              quoting=3, names= ["sentence"])
debate_democrat.shape


# In[17]:


debate_democrat.head()


# In[18]:


democrat_doc= debate_democrat["sentence"].tolist()


# In[19]:


democrat_text = tokenzied_data(democrat_doc)
print(democrat_text[0])

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
tokenized_stemmed_demo = [[p_stemmer.stem(w) for w in doc] for 
                             doc in democrat_text]
print(tokenized_stemmed_demo[0])


# In[20]:


dictionary_democrat = corpora.Dictionary(democrat_text)
print(dictionary_democrat)


# In[21]:


print(dictionary_democrat.token2id)


# In[22]:


corpus_democrat = [dictionary_democrat.doc2bow(text) for text in tokenized_stemmed_demo]
print(corpus_democrat)


# In[23]:


#With K=5
lda_model_demo = gensim.models.ldamodel.LdaModel(corpus_democrat, num_topics=5, 
                                            id2word = dictionary_democrat, 
                                            passes=20)


# In[24]:


lda_model_demo.print_topics(num_topics=5,num_words=3)


# In[25]:


#With K=10
lda_model_demo = gensim.models.ldamodel.LdaModel(corpus_democrat, num_topics=10, 
                                            id2word = dictionary_democrat, 
                                            passes=20)
lda_model_demo.print_topics(num_topics=10,num_words=3)


# In[26]:


#With K=20
lda_model_demo = gensim.models.ldamodel.LdaModel(corpus_democrat, num_topics=20, 
                                            id2word = dictionary_democrat, 
                                            passes=20)
lda_model_demo.print_topics(num_topics=20,num_words=3)


# In[27]:


topics_matrix_demo = lda_model_demo.show_topics(formatted=False, num_words=20)
topics_matrix_demo = np.array(topics_matrix_demo)

topic_words = topics_matrix_demo[:,:,1]
for i in topic_words:
    print([str(word) for word in i])
    print()


# # Republican file

# In[28]:


debate_republican = pd.read_csv("merged-republican-debate-transcripts.txt",header=None,delimiter='\n',
                              quoting=3, names= ["sentence"])
debate_republican.shape


# In[29]:


debate_republican.head()


# In[30]:


republican_doc= debate_republican["sentence"].tolist()


# In[31]:


repub_text = tokenzied_data(republican_doc)
print(repub_text)


# In[32]:


from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
tokenized_stemmed_repub = [[p_stemmer.stem(w) for w in doc] for 
                             doc in repub_text]
print(tokenized_stemmed_repub[0])


# In[33]:


dictionary_repub = corpora.Dictionary(repub_text)
print(dictionary_repub)


# In[34]:


print(dictionary_repub.token2id)


# In[35]:


corpus_republican = [dictionary_repub.doc2bow(text) for text in tokenized_stemmed_repub]
print(corpus_republican)


# In[36]:


#With K=5
lda_model_repub = gensim.models.ldamodel.LdaModel(corpus_republican, num_topics=5, 
                                            id2word = dictionary_repub, 
                                            passes=20)


# In[37]:


lda_model_repub.print_topics(num_topics=5,num_words=3)


# In[38]:


#With K=10
lda_model_repub = gensim.models.ldamodel.LdaModel(corpus_republican, num_topics=10, 
                                            id2word = dictionary_repub, 
                                            passes=20)

lda_model_repub.print_topics(num_topics=10,num_words=3)


# In[39]:


#With K=20
lda_model_repub = gensim.models.ldamodel.LdaModel(corpus_republican, num_topics=20, 
                                            id2word = dictionary_repub, 
                                            passes=20)

lda_model_repub.print_topics(num_topics=20,num_words=3)


# ### What qualitative differences might you draw here regarding the content of the debates
# 
# In order to inspect the debates, I thought to determine the topic composition and their scores. Looking at the below scores the LDA model somewhat predicted for republican debate file that the debate is about some political discussion. But do not actually predicted for democrat debate file.  
# 

# In[48]:


#Democrat file:

for index, score in sorted(lda_model_demo[corpus_democrat[0]], key=lambda tup: -1*tup[1]):
    print ("Score: {}\t Topic: {}".format(score, lda_model_demo.print_topic(index, 10)))


# In[50]:


#Republican file:

for index, score in sorted(lda_model_repub[corpus_republican[0]], key=lambda tup: -1*tup[1]):
    print ("Score: {}\t Topic: {}".format(score, lda_model_repub.print_topic(index, 10)))

