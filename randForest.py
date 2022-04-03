# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# random forest
from sklearn.ensemble import RandomForestClassifier

def rf_pred(X_scaled, y_train, max_depth):

	# Setting RF Model
	clf = RandomForestClassifier(max_depth = max_depth, random_state=0)
	clf.fit(X_scaled, y_train)

	test_score = clf.score(X_scaled, y_train)

	# Cross Validating
	cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

	# create model
	model = RandomForestClassifier(max_depth = max_depth, random_state=0)

	# evaluate model
	scores = cross_val_score(model, X_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

	# report performance
	print('Accuracy on test data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))

	return clf