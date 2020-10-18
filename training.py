from sklearn.model_selection import cross_val_score

from models import *
from visualization import read_model, save_model

XGB_NAME = 'models/best_xgb.pkl'
SVM_NAME = 'models/best_svm.pkl'
GRBOOST_NAME = 'models/best_gbc.pkl'
KNN_NAME = 'models/best_knn.pkl'
RANDFOR_NAME = 'models/best_randfor.pkl'
ADA_BOOST = 'models/best_ada.pkl'
EXTRE_NAME = 'models/best_extre.pkl'


def find_best_model(X_val, y_val):
    # ----------------------- Evaluate Models -----------------
    #xgboost(X_val, y_val)
    svm(X_val, y_val)
    #rand_forest(X_val, y_val)
    #knn(X_val, y_val)
    #dec_tree(X_val, y_val)
    extratrees(X_val, y_val)


def train(train_x, train_y, val_x, val_y, test):
    # Define best models (with hyperparameter tuning)
    find_best_model(val_x, val_y)

    # Load best models
    model1 = read_model(SVM_NAME)
    model2 = read_model(EXTRE_NAME)

    # Ensemble modeling
    model = VotingClassifier(estimators=[(
        'm1', model1), ('m2', model2)], voting='soft', n_jobs=-1)

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
