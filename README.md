# Automatic classifier of Amazon commentaries

This project aims to classify the commentaries POSITIVE, NEUTRAL of NEGATIVE found on Amazon.


### Why
I wanted to realize a classifier based on strings. Amazon provides a lot of data, I have chosen to realize a classifier with Scikit-Learn. One might think that a neural network may be more adapted.

### Input data
The data can be loaded at the following link :

### Project Phases
After loading the data and splitting them into train_set and test_test. We can apply CountVectorizer and have the occurence of each word in a DataFrame.
I have chosen 4 classifier. We can observe from the confusion matrix, that the model is good at predicting the NEGATIVE, and not to predict the POSITIVE comments. The reason is that the data set is unbalanced. I have chosen to undersample the majority class.
