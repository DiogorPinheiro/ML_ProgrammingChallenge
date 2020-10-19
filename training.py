from sklearn.model_selection import cross_val_score

from models import *
from visualization import read_model, save_model, compare_models
import pandas as pd

XGB_NAME = 'models/best_xgb.pkl'
SVM_NAME = 'models/best_svm.pkl'
GRBOOST_NAME = 'models/best_gbc.pkl'
KNN_NAME = 'models/best_knn.pkl'
RANDFOR_NAME = 'models/best_randfor.pkl'
ADA_NAME = 'models/best_ada.pkl'
EXTRE_NAME = 'models/best_extre.pkl'
NAIVE_NAME = 'models/best_naive.pkl'


def find_best_model(X_val, y_val):
    # ----------------------- Evaluate Models -----------------
    xgboost(X_val, y_val)
    svm(X_val, y_val)
    rand_forest(X_val, y_val)
    knn(X_val, y_val)
    dec_tree(X_val, y_val)
    extratrees(X_val, y_val)
    adaboost(X_val, y_val)
    naive_bayes(X_val, y_val)
    grad_boost(X_val, y_val)


def model_comparison(models, train_x, train_y):
    mean_res = []
    std_res = []
    for model in models:
        accuracy = cross_val_score(
            model, train_x, train_y, scoring='accuracy', cv=10)
        mean_res.append(accuracy.mean())
        std_res.append(accuracy.std())

    results = pd.DataFrame({"CrossValMeans": mean_res, "CrossValerrors": std_res, "Classifier": ["SVM", "ExtraTrees", "AdaBoost",
                                                                                                 "XGBoost",  "GradientBoosting", "NaiveBayes", "KNeighbours", "RandForest"]})
    compare_models(results, std_res)


def train(train_x, train_y, val_x, val_y, test):
    # Define best models (with hyperparameter tuning)
    #find_best_model(val_x, val_y)

    # Load best models
    model1 = read_model(SVM_NAME)
    model2 = read_model(EXTRE_NAME)
    model3 = read_model(ADA_NAME)
    model4 = read_model(XGB_NAME)
    model5 = read_model(GRBOOST_NAME)
    model6 = read_model(NAIVE_NAME)
    model7 = read_model(KNN_NAME)
    model8 = read_model(RANDFOR_NAME)

    models = [model1, model2, model3, model4, model5, model6, model7, model8]
    # Plot cross-validation accuracy difference between classifiers
    model_comparison(models, train_x, train_y)

    # Ensemble modeling
    model = VotingClassifier(
        estimators=[('m2', model2), ('m4', model4), ('m5', model5), ('m1', model1), ('m8', model8)], voting='soft', n_jobs=-1)

    # Save Model
    save_model(model, 'best_model.pkl')

    # Fit training data
    best_model = model.fit(train_x, train_y)

    # Evaluate cross_val_score
    accuracy = cross_val_score(
        model, train_x, train_y, scoring='accuracy', cv=10)
    print("Accuracy of best model in training : {}".format(accuracy.mean() * 100))

    # Predict and return output classes
    output_classes = best_model.predict(test)
    return output_classes
