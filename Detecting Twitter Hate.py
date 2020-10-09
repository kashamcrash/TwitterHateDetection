#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Build a model to identify hate speech (racist or sexist tweets) in Twitter
# Credentials: kasham1991@gmail.com / Karan Sharma


# Agenda
# 1. Cleanup specific parts of tweets data
# 2. Create a classification model by using Logistic Regression
# 3. Use regularization, hyperparameter tuning, stratified k-fold and cross-validation to get the best model


# In[2]:


# Importing the initial libraries 
import pandas as pd
import numpy as np
import os
import re


# In[3]:


# Loading the dataset
tweets = pd.read_csv('C://Datasets//TwitterHate.csv')
tweets.head()
# tweets.info()


# In[4]:


# Looking at the total number of labels
# A highly unbalanced dataset
# tweets.label.value_counts()
tweets.label.value_counts(normalize=True)


# In[5]:


tweets.tweet.sample().values[0]


# In[6]:


# Getting the tweets into a list 
# The tweets contains @user id handles, hashtags, url links, etc
tweet_list = tweets.tweet.values
# len(tweet_list)
tweet_list[:5]


# In[7]:


# Cleaning the tweet list - Step by Step
# 1. Normalize the casing
# 2. Using regular expressions, remove user handles. These begin with '@’
# 3. Using regular expressions, remove URLs
# 4. Using TweetTokenizer from NLTK, tokenize the tweets into individual terms
# 5. Remove stop words.
# 6. Remove redundant terms like ‘amp’, ‘rt’, etc
# 7. Remove ‘#’ symbols from the tweet while retaining the term
import re


# In[8]:


# Normalizing the casing to lower
lower_tweets = [twt.lower() for twt in tweet_list]
lower_tweets[:5]


# In[9]:


# Removing @
# re.sub("@\w+","", "@chocolate is the best! http://rahimbaig.com/ai")
no_user = [re.sub("@\w+","", twt) for twt in lower_tweets]
no_user[:5]


# In[10]:


# Removing url links
# re.sub("\w+://\S+","", "@chocolate is the best! http://rahimbaig.com/ai")
no_url = [re.sub("\w+://\S+","", twt) for twt in no_user]
no_url[:5]


# In[11]:


# Tokenization
from nltk.tokenize import TweetTokenizer
token = TweetTokenizer()
# print(token.tokenize(no_url[0]))
final_token = [token.tokenize(sent) for sent in no_url]
print(final_token[0])


# In[12]:


from nltk.corpus import stopwords
from string import punctuation

stop_nltk = stopwords.words("english")
stop_punct = list(punctuation)
stop_punct.extend(['...','``',"''",".."])
stop_context = ['rt', 'amp']
stop_final = stop_nltk + stop_punct + stop_context


# In[13]:


# Creating a function for removing terms with lenght = 1

def Remover(sent):
    return [re.sub("#","",term) for term in sent if ((term not in stop_final) & (len(term)>1))]

Remover(final_token[0])


# In[14]:


# Final set of tweets
clean_tweets = [Remover(tweet) for tweet in final_token]
clean_tweets[:5]


# #### Check out the top terms in the tweets

# In[15]:


# Looking for the top terms
# Creating an emply list and putting the top values in it
from collections import Counter
top_terms = []
for tweet in clean_tweets:top_terms.extend(tweet)


# In[16]:


toppr = Counter(top_terms)
toppr.most_common(10)


# In[17]:


# Preparing the cleaned data for modeling
# Converting tokens into strings
clean_tweets[0]


# In[18]:


clean_tweets = [" ".join(tweet) for tweet in clean_tweets]
clean_tweets[0]


# In[19]:


# Splitting the data 70/30
from sklearn.model_selection import train_test_split
x = clean_tweets
y = tweets.label.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)


# In[20]:


# Creating TFIDF and BOW
# Maximum of 5000 terms in your vocabulary
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(max_features = 5000)

# Fitting on the train/test data
x_train_bow = vector.fit_transform(x_train)
x_test_bow = vector.transform(x_test)
# x_train_bow.shape, x_test_bow.shape


# In[21]:


# Model building: Ordinary Logistic Regression
from sklearn.linear_model import LogisticRegression

logger = LogisticRegression()
logger.fit(x_train_bow, y_train)


# In[22]:


# Predicting on train/test
y_train_pred = logger.predict(x_train_bow)
y_test_pred = logger.predict(x_test_bow)
# y_test_pred 
# y_train_pred


# In[23]:


#Classification report for model evaluation
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_train, y_train_pred)


# In[24]:


print(classification_report(y_train, y_train_pred))


# In[38]:


# Adjusting the class imbalance 
logger2 = LogisticRegression(class_weight = "balanced")
logger2.fit(x_train_bow, y_train)


# In[39]:


# Fitting on the data again
y_train_pred1 = logger2.predict(x_train_bow)
y_test_pred1 = logger2.predict(x_test_bow)


# In[40]:


# Revised acccuracy score
accuracy_score(y_train, y_train_pred)


# In[41]:


print(classification_report(y_train, y_train_pred))


# In[29]:


# Regularization and hyperparameter tuning
# Selecting ‘C’ and ‘penalty’ parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[30]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'C': [0.01,0.1,1,10,100],
    'penalty': ["l1","l2"]
}


# In[31]:


logger3 = LogisticRegression(class_weight = "balanced")


# In[32]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = logger3, param_grid = param_grid, 
                          cv = StratifiedKFold(4), n_jobs = -1, verbose = 1, scoring = "recall" )


# In[33]:


grid_search.fit(x_train_bow, y_train)


# In[34]:


grid_search.best_estimator_


# In[35]:


# Using the best estimator from the grid search to make predictions on the test set
y_test_pred = grid_search.best_estimator_.predict(x_test_bow)
y_train_pred = grid_search.best_estimator_.predict(x_train_bow)


# In[36]:


print(classification_report(y_test, y_test_pred))


# In[37]:


# The recall on the toxic comments is 0.77
# The f1 score is 0.96 and 0.58 respectively
# Thank You

