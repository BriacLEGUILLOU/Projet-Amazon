# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 06:33:16 2020

@author: Briac Le Guillou

Automatic classifier of Amazon commentaries
"""

import random
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class Review:
    """ Permet de facilement obtenir le texte et le score de la review"""
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
    
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
    
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    
    def even_distribute(self):
        """ we will filter our reviews
        We will make sure that nb of positive = nb of negative
        """
        negative = list(filter(lambda x:x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x:x.sentiment == Sentiment.POSITIVE, self.reviews))
        
        #print(negative[0].text)
        #print(len(negative))
        #print(len(positive))
        # On va faire baisser le nombre de positif
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

import json


###########################################
# Load the data
###########################################

# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('C:/Users/BL80FB0N/Documents/03 - Programmation/Python/Data/Amazon commentary classifier')
os.listdir()

file_name = 'Books_small.json'

# 1ère version
reviews = []
with open(file_name) as f:
    for line in f:
        #print(line)
        #print(type(line))
        review = json.loads(line)
        #print(review['reviewText'])
        #print(review['overall'])
        reviews.append((review['reviewText'], review['overall']))
reviews[5]         # we print the review number 5
reviews[5][0]      # we print the comments
reviews[5][1]      # we print the note


# 2ème version avec classe : plus facile à lire pour un utilisateur extérieur
reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

reviews[5].score
reviews[5].text

""" Les modèles ont besoin de données numériques.
Nous allons convertir les textes en nombres, puis utiliser backwords"""
###########################################
# Prep data
###########################################
len(reviews)

#from sklearn.model_selection import train_test_split
#training, test = train_test_split(reviews, test_size=0.33, random_state=42)
n = len(reviews)
training = reviews[:int(0.67 * n)]
test = reviews[int(0.67 * n):]

train_container = ReviewContainer(training)
train_container.even_distribute()   #filter object
len(train_container.reviews)  #2* le nombre de négatif


test_container = ReviewContainer(test)
test_container.even_distribute()   #filter object

#train_x = [x.text for x in training]
#train_y = [x.sentiment for x in training]
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

#test_x = [x.text for x in test]
#test_y = [x.sentiment for x in test]
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

train_x[0]
train_y[0]
""" train_container.even_distribute() va permettre d'avoir autant de positif que de négatif 
"""
print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))
# C'est bien le cas

###########################################
# Bags of words vectorisation
###########################################
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
train_x_vectors =  vectorizer.fit_transform(train_x).toarray()
test_x_vectors = vectorizer.transform(test_x).toarray()

print(train_x[0])
print(train_x_vectors[0])


###########################################
# Classification
###########################################
# Support Vector Machine
from sklearn import svm
clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)
clf_svm.predict(train_x_vectors[0])


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()

clf_dec.fit(train_x_vectors, train_y)
clf_dec.predict(train_x_vectors[0])


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors, train_y)
clf_gnb.predict(train_x_vectors[0])


# Naive Bayes
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)
clf_log.predict(train_x_vectors[0])


###########################################
# Evaluation
###########################################
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors, test_y))
print(clf_log.score(test_x_vectors, test_y))

""" Les modèles ont l'air bons, mais ce qui nous intéresse c'est le F1 score.
"""

# F1 score
from sklearn.metrics import f1_score

f1_score(test_y, clf_dec.predict(test_x_vectors), average = None, 
         labels =[Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE])
""" Very goo for positive, mauvaise pour les autres
Ici, nous sommes dans le cas d'un imbalanced dataset 
le modèle ne fait que produire positive"""


train_y.count(Sentiment.POSITIVE)
train_y.count(Sentiment.NEGATIVE)

""" We need to have an even distribution"""


""" Après avoir rendu training et test égales en positifs et négatifs,
le f1 score est maintenant le même entre postif et négatif ! """
""" Now what can we do to improve the model ?
CountVectorizer considère chaque mot avec le même poids.

This book is great !
This book was so bad

Au lieu d'utiliser CountVectorizer, on peut utiliser TfidfVectorizer 
qui va prendre en compte les fréquences des mots

Ainsi un mot aura moins d'importance s'il intervient dans beaucoup de commentaires
"""



###########################################
# Turning our model (with Grid Search)
###########################################
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1,1,8,16,32]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x, train_y)


clf.score(test_x, test_y)



###########################################
# Pickl model
###########################################
import pickle 

#Save classifier
with open('./models/category_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
    
with open('./models/category_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

#Load Classifier
with open('./models/category_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('./models/category_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)



