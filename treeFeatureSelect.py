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

def treeFeatureSelection(X_scaled, y_train, FeatureAmount,X): 
    #getting radom forest without feature selection 
    # Setting RF Model
    clf = RandomForestClassifier(n_estimators = 100, random_state=0)
    clf.fit(X_scaled, y_train)

    test_score = clf.score(X_scaled, y_train)

    # Cross Validating
    cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

    # create model
    model = RandomForestClassifier(n_estimators = 100, random_state=0)

    # evaluate model
    scores = cross_val_score(model, X_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    # reporting performance on full tree
    print('Accuracy of full random forest on test data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))


    # get the importance of the resulting features.
    importances = clf.feature_importances_
    # create a data frame for visualization.
    final_df = pd.DataFrame({"Features": X.columns, "Importances":importances})
    final_df.set_index('Importances')

    # sort in ascending order to better visualization.
    final_df = final_df.sort_values('Importances')

    # plot the feature importances in bars. 
    plt.figure(figsize=(100,20))
    plt.xticks(rotation=45)
    sns.barplot(x="Features",y= "Importances", data=final_df)


    model_tree = RandomForestClassifier(n_estimators=100,random_state=42)

    # use RFE to eleminate the less importance features
    sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=FeatureAmount, step=1)
    X_train_rfe_tree = sel_rfe_tree.fit_transform(X_scaled, y_train)


    #Reduce X to the selected features and then predict using the predict
    test_score = sel_rfe_tree.score(X_scaled, y_train)

    # Cross Validating
    cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

    # create model
    model = RandomForestClassifier(n_estimators = 100, random_state=42)
    rfe_tree = RFE(estimator=model, n_features_to_select=FeatureAmount, step=1)

    # evaluate model
    scores = cross_val_score(rfe_tree, X_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    # report performance
    print('Accuracy of feature selected random forest on test data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))

    return sel_rfe_tree




