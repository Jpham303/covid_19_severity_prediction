# Needed libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def knn_pred(X_train_selected_ANOVA, y_train, neighbors):

	neigh = KNeighborsClassifier(n_neighbors = neighbors)
	neigh.fit(X_train_selected_ANOVA, y_train)

	neigh.score(X_train_selected_ANOVA, y_train)

	# prepare the cross-validation procedure
	cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

	# create model
	model = KNeighborsClassifier(n_neighbors = neighbors)

	# evaluate model
	scores = cross_val_score(model, X_train_selected_ANOVA, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

	# report performance
	print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))