from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.svm import SVC
from scipy import stats
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from visualization import model_result, save_model, plot_learning_curve


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
    cv = StratifiedKFold(n_splits=5, random_state=1)

    gb = GradientBoostingClassifier(warm_start=True)

    # plot_learning_curve(gb, "Grad Boost",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    gb_param_grid = {'loss': ["deviance"],              # NEW
                     'n_estimators': Integer(100, 200, 300),
                     'learning_rate': Real(0.001, 0.2),
                     'max_depth': Integer(2, 8),
                     'min_samples_leaf': Integer(50, 150),
                     'max_features': Real(0.1, 0.5)
                     }
    clf_gbc = BayesSearchCV(gb, gb_param_grid, n_iter=100,
                            cv=cv, verbose=True, n_jobs=-1)
    # clf_gbc = GridSearchCV(gb, param_grid=gb_param_grid,
    #                       cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

    best_clf_gbc = clf_gbc.fit(data_x, data_y)
    model_result(best_clf_gbc, data_x, data_y, best_clf_gbc.best_params_)
    save_model(best_clf_gbc, 'models/best_gbc.pkl')
    # return best_clf_gbc.best_estimator_

# -------------------- kNN -----------------------------------


def knn(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    knn = KNeighborsClassifier()

    # plot_learning_curve(knn, "KNN",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    param_grid = {'n_neighbors': [3, 5, 7, 9, 12, 15, 18, 21],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'p': [1, 2]}
    clf_knn = GridSearchCV(knn, param_grid=param_grid,
                           cv=cv, verbose=True, n_jobs=-1)
    best_clf_knn = clf_knn.fit(data_x, data_y)
    model_result(best_clf_knn, data_x, data_y, best_clf_knn.best_params_)
    save_model(best_clf_knn, 'models/best_knn.pkl')
    # return best_clf_knn.best_estimator_


# -------------------- Random Forest -------------------------


def rand_forest(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    rf = RandomForestClassifier(random_state=1)

    # plot_learning_curve(rf, "RandForest",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    param_grid = {'n_estimators': Integer(100, 550),  # NEW
                  'criterion': Categorical(['gini', 'entropy']),
                  'bootstrap': Categorical([True]),
                  'max_depth': Integer(3, 25),
                  'max_features': Categorical(['auto', 'sqrt', 'log2']),
                  'min_samples_leaf': Integer(2, 5),
                  'min_samples_split': Integer(2, 5)
                  }
    clf_rf = BayesSearchCV(rf, param_grid, n_iter=100,
                           cv=cv, verbose=True, n_jobs=-1)
    # clf_rf = GridSearchCV(rf, param_grid=param_grid,
    #                      cv=cv, verbose=True, n_jobs=-1)
    best_clf_rf = clf_rf.fit(data_x, data_y)
    model_result(best_clf_rf, data_x, data_y, best_clf_rf.best_params_)
    save_model(best_clf_rf, 'models/best_randfor.pkl')
    # return best_clf_rf.best_estimator_


# --------------------- SVM -----------------------------------


def svm(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    svc = SVC(probability=True)

    # plot_learning_curve(svc, "SVM",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    param_grid = tuned_parameters = [{'kernel': Categorical(['rbf']), 'gamma': Real(.1, 10),
                                      'C': Real(.1, 1000)}]
    # param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10],
    #                                  'C': [.1, 1, 10, 100, 1000]}]
    clf_svc = BayesSearchCV(svc, param_grid, n_iter=100,
                            cv=cv, verbose=True, n_jobs=-1)
    # clf_svc = RandomizedSearchCV(
    #    svc, param_distributions=param_grid, n_iter=100, cv=cv, verbose=True, n_jobs=-1)
    # clf_svc = GridSearchCV(svc, param_grid=param_grid,
    #                       cv=5, verbose=True, n_jobs=-1)
    best_clf_svc = clf_svc.fit(data_x, data_y)
    model_result(best_clf_svc, data_x, data_y, best_clf_svc.best_params_)
    save_model(best_clf_svc, 'models/best_svm.pkl')
    # return best_clf_svc.best_estimator_

# --------------------- xgBoost -------------------------------


def xgboost(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    xgb = XGBClassifier(random_state=1, objective='multi: softmax')

    # plot_learning_curve(xgb, "xgb",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    # param_grid = {'n_estimators': stats.randint(150, 500),
    #              'learning_rate': stats.uniform(0.01, 0.07),
    #              'subsample': stats.uniform(0.3, 0.7),
    #              'max_depth': [3, 4, 5, 6, 7, 8, 9],
    #              'colsample_bytree': stats.uniform(0.5, 0.45),
    #              'min_child_weight': [1, 2, 3]
    #              }
    param_grid = {'n_estimators': Integer(50, 500),  # NEW
                  'learning_rate': Real(0.01, 0.15),
                  'subsample': Real(0.3, 0.7),
                  'max_depth': Integer(3, 9),
                  'colsample_bytree': Real(0.45, 0.5),
                  'min_child_weight': Integer(1, 3)
                  }
    clf_xgb = BayesSearchCV(xgb, param_grid, n_iter=100,
                            cv=cv, verbose=True, n_jobs=-1)
    # clf_xgb = RandomizedSearchCV(
    #    xgb, param_distributions=param_grid, n_iter=100, cv=cv, verbose=True, n_jobs=-1)
    # clf_xgb = GridSearchCV(xgb, param_grid=param_grid,
    #                       cv=5, verbose=True, n_jobs=-1)
    best_clf_xgb = clf_xgb.fit(data_x, data_y)
    model_result(best_clf_xgb, data_x, data_y, best_clf_xgb.best_params_)
    save_model(best_clf_xgb, 'models/best_xgb.pkl')
    # return best_clf_xgb.best_estimator_


# -------------------------- Adaboost ---------------------------------


def adaboost(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    model = AdaBoostClassifier(base_estimator=ExtraTreesClassifier())

    # plot_learning_curve(model, "AdaBoost",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    grid_search = GridSearchCV(
        estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    best_clf_ada = grid_search.fit(data_x, data_y)
    model_result(best_clf_ada, data_x, data_y, best_clf_ada.best_params_)
    save_model(best_clf_ada, 'models/best_ada.pkl')
    return best_clf_ada.best_estimator_

# -------------------- Extra Trees --------------------------------------


def extratrees(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    ext = ExtraTreesClassifier()

    # plot_learning_curve(ext, "ExtraTrees",
    #                    data_x, data_y, cv=cv, n_jobs=-1)   # Plot learning curve

    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}
    gsExtC = GridSearchCV(ext, param_grid=ex_param_grid,
                          cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    best_clf_extre = gsExtC.fit(data_x, data_y)
    model_result(best_clf_extre, data_x, data_y, best_clf_extre.best_params_)
    save_model(best_clf_extre, 'models/best_extre.pkl')
    # return best_clf_extre.best_estimator_

# --------------------- Light GBM ------------------------------------


def lightgb(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    xlb = LGBMClassifier()
    param_grid = {'n_estimators': Integer(50, 500),
                  'learning_rate': Real(0.001, 0.15),
                  'subsample': Real(0.3, 0.7),
                  'max_depth': Integer(3, 9),
                  'min_child_weight': Real(0.001, 3.0)
                  }
    clf_lgb = BayesSearchCV(xlb, param_grid, n_iter=100,
                            cv=cv, verbose=True, n_jobs=-1)
    best_clf_lgb = clf_lgb.fit(data_x, data_y)
    model_result(best_clf_lgb, data_x, data_y, best_clf_lgb.best_params_)
    save_model(best_clf_lgb, 'models/best_lgb.pkl')
    # return best_clf_lgb.best_estimator_

# -------------------------- CatBoost ----------------------------------


def catbo(data_x, data_y):
    cv = StratifiedKFold(n_splits=5, random_state=1)

    ctb = CatBoostClassifier(verbose=0)
    param_grid = {'iterations': Integer(10, 1000),
                  'depth': Integer(1, 8),
                  'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                  'random_strength': Real(1e-9, 10, 'log-uniform'),
                  'bagging_temperature': Real(0.0, 1.0),
                  'border_count': Integer(1, 255),
                  'l2_leaf_reg': Integer(2, 30),
                  'n_estimators': Integer(50, 200),
                  'scale_pos_weight': Real(0.01, 1.0, 'uniform')}

    clf_ctb = BayesSearchCV(ctb, param_grid, n_iter=100,
                            cv=cv, verbose=True, n_jobs=-1)  # USe 1 job to avoid segmentation
    best_clf_ctb = clf_ctb.fit(data_x, data_y)
    model_result(best_clf_ctb, data_x, data_y, best_clf_ctb.best_params_)
    save_model(best_clf_ctb, 'models/best_ctb.pkl')
    # return best_clf_ctb.best_estimator_
