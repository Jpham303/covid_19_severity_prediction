# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# KNN
from sklearn.neighbors import KNeighborsClassifier

def knn_pred(X_train_selected_ANOVA, y_train, neighbors):

	neigh = KNeighborsClassifier(n_neighbors = neighbors).fit(X_train_selected_ANOVA, y_train)

	test_score = neigh.score(X_train_selected_ANOVA, y_train)

	# prepare the cross-validation procedure
	cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

	# create model
	model = KNeighborsClassifier(n_neighbors = neighbors)

	# evaluate model
	scores = cross_val_score(model, X_train_selected_ANOVA, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

	# report performance
	print('Accuracy on test data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))

	return neigh