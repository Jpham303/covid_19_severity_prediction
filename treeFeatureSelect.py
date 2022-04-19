# Needed libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Tree Feature Selection 
from sklearn.ensemble import RandomForestClassifier 
# Import the RFE from sklearn library
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier 
# Import the RFE from sklearn library
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

# evaluate a model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def treeFeatureSelection(X_scaled, y_train, FeatureAmount, max_depth = None): 
    model = RandomForestClassifier(n_estimators=68, random_state=0, max_depth = max_depth)

    # use RFE to eleminate the less importance features
    sel_rfe_tree = RFE(estimator = model, n_features_to_select = FeatureAmount, step = 1)
    sel_rfe_tree.fit_transform(X_scaled, y_train)

    #Reduce X to the selected features and then predict using the predict
    test_score = sel_rfe_tree.score(X_scaled, y_train)

    # create model
    model = RandomForestClassifier(n_estimators = 68, random_state=0, max_depth = max_depth)
    rfe_tree = RFE(estimator = model, n_features_to_select = FeatureAmount, step = 1)

    # evaluate model
    scores = cross_val_score(rfe_tree, X_scaled, y_train, cv = 10, n_jobs=-1)

    # report performance
    print('Accuracy of feature selected random forest on train data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))

    return sel_rfe_tree




