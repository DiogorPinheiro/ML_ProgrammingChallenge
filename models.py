from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.svm import SVC
from scipy import stats

from visualization import model_result

# ------------------ Logistic Regression with L1 Regularization ------------------------


# def log_reg(data_x, data_y):

# ------------------- Naive Bayes -----------------------------


def naive_bayes(data_x, data_y):
    gnb = GaussianNB()
    cv = cross_val_score(gnb, data_x, data_y, cv=5)
    print(cv.mean())

# -------------------- Decision Tree -------------------------


def dec_tree(data_x, data_y):
    dt = tree.DecisionTreeClassifier(random_state=1)
    cv = cross_val_score(dt, data_x, data_y, cv=5)
    print(cv.mean())

# -------------------- kNN -----------------------------------


def knn(data_x, data_y):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 9, 12, 15, 18, 21],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'p': [1, 2]}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    clf_knn = GridSearchCV(knn, param_grid=param_grid,
                           cv=cv, verbose=True, n_jobs=-1)
    best_clf_knn = clf_knn.fit(data_x, data_y)
    model_result(best_clf_knn)


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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    clf_rf = GridSearchCV(rf, param_grid=param_grid,
                          cv=cv, verbose=True, n_jobs=-1)
    best_clf_rf = clf_rf.fit(data_x, data_y)
    model_result(best_clf_rf)


# --------------------- SVM -----------------------------------


def svm(data_x, data_y):
    svc = SVC(probability=True)
    param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10],
                                      'C': [.1, 1, 10, 100, 1000]},
                                     {'kernel': ['linear'],
                                         'C': [.1, 1, 10, 100, 1000]},
                                     {'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'C': [.1, 1, 10, 100, 1000]}]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    clf_svc = RandomizedSearchCV(
        svc, param_distributions=param_grid, n_iter=100, cv=cv, verbose=True, n_jobs=-1)
    # clf_svc = GridSearchCV(svc, param_grid=param_grid,
    #                       cv=5, verbose=True, n_jobs=-1)
    best_clf_svc = clf_svc.fit(data_x, data_y)
    model_result(best_clf_svc)
# --------------------- xgBoost -------------------------------


def xgboost(data_x, data_y):
    xgb = XGBClassifier(random_state=1, objective='multi: softmax')

    param_grid = {'n_estimators': stats.randint(150, 500),
                  'learning_rate': stats.uniform(0.01, 0.07),
                  'subsample': stats.uniform(0.3, 0.7),
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'colsample_bytree': stats.uniform(0.5, 0.45),
                  'min_child_weight': [1, 2, 3]
                  }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    clf_xgb = RandomizedSearchCV(
        xgb, param_distributions=param_grid, n_iter=100, cv=cv, verbose=True, n_jobs=-1)
    # clf_xgb = GridSearchCV(xgb, param_grid=param_grid,
    #                       cv=5, verbose=True, n_jobs=-1)
    best_clf_xgb = clf_xgb.fit(data_x, data_y)
    model_result(best_clf_xgb)

# -------------------------- Adaboost ---------------------------------


def adaboost(data_x, data_y):
    model = AdaBoostClassifier()
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(
        estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    grid_result = grid_search.fit(data_x, data_y)
    model_result(grid_result)
