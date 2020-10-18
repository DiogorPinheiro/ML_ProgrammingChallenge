from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.svm import SVC
from scipy import stats
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from visualization import model_result, save_model

# ------------------ Logistic Regression with L1 Regularization ------------------------


# def log_reg(data_x, data_y):

# ------------------- Naive Bayes -----------------------------


def naive_bayes(data_x, data_y):
    gnb = GaussianNB()
    cv = cross_val_score(gnb, data_x, data_y, cv=5)
    nb = gnb.fit(data_x, data_y)
    print(cv.mean())
    save_model(nb, 'models/best_naive.pkl')

# -------------------- Decision Tree -------------------------


def dec_tree(data_x, data_y):
    dt = tree.DecisionTreeClassifier(random_state=1)
    cv = cross_val_score(dt, data_x, data_y, cv=5)
    print(cv.mean())

# ---------------------- Gradient Boosting -------------------


def grad_boost(data_x, data_y):
    gb = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf_gbc = GridSearchCV(gb, param_grid=gb_param_grid,
                           cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

    best_clf_gbc = clf_gbc.fit(data_x, data_y)
    model_result(best_clf_gbc)
    save_model(best_clf_gbc.best_estimator_, 'models/best_gbc.pkl')
    # return best_clf_gbc.best_estimator_

# -------------------- kNN -----------------------------------


def knn(data_x, data_y):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 9, 12, 15, 18, 21],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'p': [1, 2]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf_knn = GridSearchCV(knn, param_grid=param_grid,
                           cv=cv, verbose=True, n_jobs=-1)
    best_clf_knn = clf_knn.fit(data_x, data_y)
    model_result(best_clf_knn)
    save_model(best_clf_knn.best_estimator_, 'models/best_knn.pkl')
    # return best_clf_knn.best_estimator_


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
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf_rf = GridSearchCV(rf, param_grid=param_grid,
                          cv=cv, verbose=True, n_jobs=-1)
    best_clf_rf = clf_rf.fit(data_x, data_y)
    model_result(best_clf_rf)
    save_model(best_clf_rf.best_estimator_, 'models/best_randfor.pkl')
    # return best_clf_rf.best_estimator_


# --------------------- SVM -----------------------------------


def svm(data_x, data_y):
    svc = SVC(probability=True)
    param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10],
                                      'C': [.1, 1, 10, 100, 1000]}]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf_svc = RandomizedSearchCV(
        svc, param_distributions=param_grid, n_iter=100, cv=cv, verbose=True, n_jobs=-1)
    # clf_svc = GridSearchCV(svc, param_grid=param_grid,
    #                       cv=5, verbose=True, n_jobs=-1)
    best_clf_svc = clf_svc.fit(data_x, data_y)
    model_result(best_clf_svc)
    save_model(best_clf_svc.best_estimator_, 'models/best_svm.pkl')
    # return best_clf_svc.best_estimator_

# --------------------- xgBoost -------------------------------


def xgboost(data_x, data_y):
    xgb = XGBClassifier(random_state=1, objective='multi: softmax')
    # param_grid = {'n_estimators': stats.randint(150, 500),
    #              'learning_rate': stats.uniform(0.01, 0.07),
    #              'subsample': stats.uniform(0.3, 0.7),
    #              'max_depth': [3, 4, 5, 6, 7, 8, 9],
    #              'colsample_bytree': stats.uniform(0.5, 0.45),
    #              'min_child_weight': [1, 2, 3]
    #              }
    param_grid = {'n_estimators': Integer(150, 500),
                  'learning_rate': Real(0.01, 0.07),
                  'subsample': Real(0.3, 0.7),
                  'max_depth': Integer(3, 9),
                  'colsample_bytree': Real(0.45, 0.5),
                  'min_child_weight': Integer(1, 3)
                  }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf_xgb = BayesSearchCV(xgb, param_grid, n_iter=100,
                            cv=cv, verbose=True, n_jobs=-1)
    # clf_xgb = RandomizedSearchCV(
    #    xgb, param_distributions=param_grid, n_iter=100, cv=cv, verbose=True, n_jobs=-1)
    # clf_xgb = GridSearchCV(xgb, param_grid=param_grid,
    #                       cv=5, verbose=True, n_jobs=-1)
    best_clf_xgb = clf_xgb.fit(data_x, data_y)
    model_result(best_clf_xgb)
    save_model(best_clf_xgb.best_estimator_, 'models/best_xgb.pkl')
    # return best_clf_xgb.best_estimator_


# -------------------------- Adaboost ---------------------------------


def adaboost(data_x, data_y):
    model = AdaBoostClassifier()
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(
        estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    best_clf_ada = grid_search.fit(data_x, data_y)
    model_result(best_clf_ada)
    save_model(best_clf_ada.best_estimator_, 'models/best_ada.pkl')
    # return best_clf_ada.best_estimator_

# -------------------- Extra Trees --------------------------------------


def extratrees(data_x, data_y):
    ext = ExtraTreesClassifier()

    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    gsExtC = GridSearchCV(ext, param_grid=ex_param_grid,
                          cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    best_clf_extre = gsExtC.fit(data_x, data_y)
    model_result(best_clf_extre)
    save_model(best_clf_extre.best_estimator_, 'models/best_extre.pkl')
    # return best_clf_extre.best_estimator_
