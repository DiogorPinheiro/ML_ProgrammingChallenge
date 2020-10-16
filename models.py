from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# ------------------ Logistic Regression with L1 Regularization ------------------------


# ------------------- Naive Bayes -----------------------------
def naive_bayes(data_x, data_y):
    gnb = GaussianNB()
    cv = cross_val_score(gnb, data_x, data_y, cv=5)
    return cv.mean()

# -------------------- Decision Tree -------------------------


def dec_tree(data_x, data_y):
    dt = tree.DecisionTreeClassifier(random_state=1)
    cv = cross_val_score(dt, X_train_scaled, y_train, cv=5)
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
    best_clf_knn = clf_knn.fit(X_train_scaled, y_train)
    clf_performance(best_clf_knn, 'KNN')

# -------------------- Random Forest -------------------------


def rand_forest(data_x, data_y):
    rf = RandomForestClassifier(random_state=1)
    cv = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    return cv.mean()

# --------------------- SVM -----------------------------------


def svm(data_x, data_y):

    # --------------------- xgBoost -------------------------------


def xgboost(data_x, data_y):
    xgb = XGBClassifier(random_state=1)
    cv = cross_val_score(xgb, data_x, data_y, cv=5)
    return cv.mean()
