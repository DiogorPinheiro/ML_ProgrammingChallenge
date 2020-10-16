from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# ------------------ Logistic Regression with L1 Regularization ------------------------


# def log_reg(data_x, data_y):

# ------------------- Naive Bayes -----------------------------


def naive_bayes(data_x, data_y):
    gnb = GaussianNB()
    cv = cross_val_score(gnb, data_x, data_y, cv=5)
    return cv.mean()

# -------------------- Decision Tree -------------------------


def dec_tree(data_x, data_y):
    dt = tree.DecisionTreeClassifier(random_state=1)
    cv = cross_val_score(dt, data_x, data_y, cv=5)
    return cv.mean()

# -------------------- kNN -----------------------------------


def knn(data_x, data_y):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'p': [1, 2]}
    clf_knn = GridSearchCV(knn, param_grid=param_grid,
                           cv=5, verbose=True, n_jobs=-1)
    best_clf_knn = clf_knn.fit(data_x, data_y)
    return best_clf_knn.best_score_

# -------------------- Random Forest -------------------------


def rand_forest(data_x, data_y):
    rf = RandomForestClassifier(random_state=1)
    param_grid = {'n_estimators': [400, 450, 500, 550],
                  'criterion': ['gini', 'entropy'],
                  'bootstrap': [True],
                  'max_depth': [15, 20, 25],
                  'max_features': ['auto', 'sqrt', 10],
                  'min_samples_leaf': [2, 3],
                  'min_samples_split': [2, 3]}

    clf_rf = GridSearchCV(rf, param_grid=param_grid,
                          cv=5, verbose=True, n_jobs=-1)
    best_clf_rf = clf_rf.fit(data_x, data_y)
    return best_clf_rf.best_score_

# --------------------- SVM -----------------------------------


def svm(data_x, data_y):
    svc = SVC(probability=True)
    param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10],
                                      'C': [.1, 1, 10, 100, 1000]},
                                     {'kernel': ['linear'],
                                         'C': [.1, 1, 10, 100, 1000]},
                                     {'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'C': [.1, 1, 10, 100, 1000]}]
    clf_svc = GridSearchCV(svc, param_grid=param_grid,
                           cv=5, verbose=True, n_jobs=-1)
    best_clf_svc = clf_svc.fit(data_x, data_y)
    return best_clf_svc.best_score_
# --------------------- xgBoost -------------------------------


def xgboost(data_x, data_y):
    xgb = XGBClassifier(random_state=1)

    param_grid = {
        'n_estimators': [20, 50, 100, 250, 500, 1000],
        'colsample_bytree': [0.2, 0.5, 0.7, 0.8, 1],
        'max_depth': [2, 5, 10, 15, 20, 25, None],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 1.5, 2],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        'learning_rate': [.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        'gamma': [0, .01, .1, 1, 10, 100],
        'min_child_weight': [0, .01, 0.1, 1, 10, 100],
        'sampling_method': ['uniform', 'gradient_based']
    }

    clf_xgb = GridSearchCV(xgb, param_grid=param_grid,
                           cv=5, verbose=True, n_jobs=-1)
    best_clf_xgb = clf_xgb.fit(data_x, data_y)
    return best_clf_xgb.best_score_
