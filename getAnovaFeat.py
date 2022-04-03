from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def get_anova_feat(ks, X_scaled, y_train):
    fs = SelectKBest(score_func = f_classif, k = ks)
    X_train_selected_ANOVA = fs.fit_transform(X_scaled, y_train)
    
    return X_train_selected_ANOVA