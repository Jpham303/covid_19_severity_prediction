import pandas as pd
import matplotlib.pyplot as plt

#Tree Feature Selection 
from asyncio.windows_events import NULL
from sklearn.ensemble import RandomForestClassifier 

# Import the RFE from sklearn library
from sklearn.feature_selection import RFE
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics

# evaluate a model using k-fold cross-validation
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def rand_forest(X_scaled, y_train, X, max_depth = None):
	# getting radom forest without feature selection 
    # Setting RF Model
    # clf = RandomForestClassifier(n_estimators = 100, random_state=0)
    # clf.fit(X_scaled, y_train)

    # test_score = clf.score(X_scaled, y_train)

    # Cross Validating
    # cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

    # create model
    model = RandomForestClassifier(n_estimators = 100, random_state=0, max_depth = max_depth)
    model.fit(X_scaled, y_train)
    test_score = model.score(X_scaled, y_train)

    # evaluate model
    scores = cross_val_score(model, X_scaled, y_train, scoring='accuracy', cv=10, n_jobs=-1)

    # reporting performance on full tree
    print('Accuracy of full random forest on train data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))


    # get the importance of the resulting features.
    importances = model.feature_importances_
    # create a data frame for visualization.
    final_df = pd.DataFrame({"Features": X.columns, "Importances":importances})
    final_df.set_index('Importances')

    # sort in ascending order to better visualization.
    final_df = final_df.sort_values('Importances')

    # plot the feature importances in bars. 
    plt.figure(figsize=(100,20))
    plt.xticks(rotation=45)
    sns.barplot(x="Features",y= "Importances", data=final_df)
    
    return model