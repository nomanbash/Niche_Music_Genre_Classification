import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as ms
%matplotlib inline
import numpy as np
import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import IPython.display as ipd

# Importing required libraries
from seaborn import load_dataset, pairplot

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('features_30_sec.csv')
df = df.drop('filename', axis=1)


X = df.drop('label',axis=1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cols = X_train.columns




from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


# import SVC classifier
from sklearn.svm import SVC


# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Model accuracy score with default hyperparameters: 0.6850





##################################################
# ----------------- rbf kernel ----------------- #




# instantiate classifier with rbf kernel and C=10
svc=SVC(C=10) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=10.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with rbf kernel and C=10.0 : 0.7050





# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with rbf kernel and C=10.0 : 0.7050









##################################################
# --------------- linear kernel ---------------- #

# instantiate classifier with linear kernel and C=10.0
lin_svc10=SVC(kernel='linear', C=10.0) 


# fit classifier to training set
lin_svc10.fit(X_train, y_train)


# make predictions on test set
y_pred=lin_svc10.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=10.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with linear kernel and C=10.0 : 0.6300









# instantiate classifier with polynomial kernel and C=100.0
lin_svc100=SVC(kernel='linear', C=100.0) 


# fit classifier to training set
lin_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=lin_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with linear kernel and C=100.0 : 0.6300









##################################################
# ---------------- poly kernel ----------------- #





# instantiate classifier with polynomial kernel and C=100.0
poly_svc10=SVC(kernel='poly', C=10.0) 


# fit classifier to training set
poly_svc10.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc10.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=10.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with polynomial kernel and C=10.0 : 0.7200





# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0) 


# fit classifier to training set
poly_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=100 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with polynomial kernel and C=100 : 0.6800







##################################################
# --------------- sigmoid kernel --------------- #







# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with sigmoid kernel and C=1.0 : 0.5600







# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc10=SVC(kernel='sigmoid', C=10.0) 


# fit classifier to training set
sigmoid_svc10.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc10.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=10.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with sigmoid kernel and C=10.0 : 0.4750







# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 


# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Output: Model accuracy score with sigmoid kernel and C=100.0 : 0.4350




