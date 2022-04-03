# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Decision Tree
from sklearn import tree

def decision_tree_pred(X_train_selected_ANOVA, y_train):
    #getting tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train_selected_ANOVA, y_train)

    test_score = clf.score(X_train_selected_ANOVA, y_train)

    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # create model
    model = tree.DecisionTreeClassifier()

    # evaluate model
    scores = cross_val_score(model, X_train_selected_ANOVA, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    # report performance
    print('Accuracy on test data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))

    return clf

