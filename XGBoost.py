# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import cv

def xgboost_pred(X_train_selected_ANOVA, y_train):

	params = {
		'objective':'multi:softmax',
		'num_class':3,
		'max_depth':4,
		'learning_rate':0.1,
		'n_estimators':100
		}
	xgb_cl = xgb.XGBClassifier(**params).fit(X_train_selected_ANOVA, y_train)
	preds = xgb_cl.predict(X_train_selected_ANOVA)
	test_score = accuracy_score(y_train, preds)

	# Cross Validating
#	data_dmatrix = xgb.DMatrix(data = X_train_selected_ANOVA, label = y_train)
#	cv_params = {
#		'objective':'multi:softmax',
#		'num_class':3,
#		'colsample_bytree': 0.3,
#		'learning_rate': 0.01,
#		'max_depth': 3,
#		'alpha': 10
#		}
	#xgb_cv = xgb.cv(dtrain=data_dmatrix, params=cv_params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
	
	cv = KFold(n_splits = 10, random_state = 1, shuffle = True)
	
	# create model
	model = xgb.XGBClassifier(**params)

	# evaluate model
	scores = cross_val_score(model, X_train_selected_ANOVA, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

	# report performance
	print('Accuracy on test data: %.3f || CV Accuracy: %.3f' % (test_score , mean(scores)))

	return xgb_cl

