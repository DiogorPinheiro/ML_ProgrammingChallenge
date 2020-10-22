from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from models import *
from visualization import read_model, save_model, compare_models
import pandas as pd

# File names (best models found with hyperparameter tuning)
XGB_NAME = 'models/best_xgb.pkl'
SVM_NAME = 'models/best_svm.pkl'
GRBOOST_NAME = 'models/best_gbc.pkl'
KNN_NAME = 'models/best_knn.pkl'
RANDFOR_NAME = 'models/best_randfor.pkl'
ADA_NAME = 'models/best_ada.pkl'
EXTRE_NAME = 'models/best_extre.pkl'
NAIVE_NAME = 'models/best_naive.pkl'
LGB_NAME = 'models/best_lgb.pkl'
CTB_NAME = 'models/best_ctb.pkl'


def find_best_model(X_val, y_val):
    # Hyperparameter tuning
    xgboost(X_val, y_val)
    svm(X_val, y_val)
    rand_forest(X_val, y_val)
    knn(X_val, y_val)
    dec_tree(X_val, y_val)
    extratrees(X_val, y_val)
    adaboost(X_val, y_val)
    naive_bayes(X_val, y_val)
    grad_boost(X_val, y_val)
    lightgb(X_val, y_val)
    # catbo(X_val, y_val)


def model_comparison(models, train_x, train_y):
    # Plot bar chart to compare model accuracies
    mean_res = []
    std_res = []
    for model in models:
        accuracy = cross_val_score(
            model, train_x, train_y, scoring='accuracy', cv=10)
        mean_res.append(accuracy.mean())
        std_res.append(accuracy.std())

    results = pd.DataFrame({"CrossValMeans": mean_res, "CrossValerrors": std_res, "Classifier": ["SVM", "ExtraTrees", "AdaBoost",
                                                                                                 "XGBoost",  "GradientBoosting", "NaiveBayes", "KNeighbours", "RandForest", "lightGB"]})
    compare_models(results, std_res)


def tune_weights(clf1, clf2, clf3, val_x, val_y):
    # Adapted from https://sebastianraschka.com/Articles/2014_ensemble_classifier.html#ensembleclassifier---tuning-weights
    df = []

    i = 0
    for w1 in range(1, 4):
        for w2 in range(1, 4):
            for w3 in range(1, 4):

                if len(set((w1, w2, w3))) == 1:  # skip if all weights are equal
                    continue

                eclf = VotingClassifier(
                    estimators=[clf1, clf2, clf3], weights=[w1, w2, w3], voting='soft', n_jobs=-1)
                scores = cross_val_score(
                    estimator=eclf,
                    X=val_x,
                    y=val_y,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1)

                df.append([w1, w2, w3, scores.mean(), scores.std()])
                i += 1

    print(df)


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
    model9 = read_model(LGB_NAME)
    # model10 = read_model(CTB_NAME)

    models = [model1, model2, model3, model4,
              model5, model6, model7, model8, model9]
    # Plot cross-validation accuracy difference between classifiers
    # model_comparison(models, train_x, train_y)

    # tune_weights(('m2', ExtraTreesClassifier(**model2.best_params_)), ('m4', XGBClassifier(**model4.best_params_)),
    #             ('m6', LGBMClassifier(**model9.best_params_)),  train_x, train_y)

    # Ensemble modeling
    model = VotingClassifier(
        estimators=[('m2', ExtraTreesClassifier(**model2.best_params_)), ('m4', XGBClassifier(**model4.best_params_)), ('m6', LGBMClassifier(**model9.best_params_))], voting='soft', n_jobs=-1, weights=[3, 3, 2])

    # Save Model
    save_model(model, 'best_model.pkl')

    # Fit training data
    model = model.fit(train_x, train_y)

    # Evaluate cross_val_score
    cv = StratifiedKFold(n_splits=5, random_state=1)
    accuracy = cross_val_score(
        model, train_x, train_y, scoring='accuracy', cv=cv)
    print("Accuracy of best model in training : {}".format(accuracy.mean() * 100))

    #out = model.predict_proba(val_x)
    #print(log_loss(val_y, out, labels=model.classes_))
    #print(model.score(val_x, val_y))

    # Predict and return output classes
    output_classes = model.predict(test)
    return output_classes
