import pickle
from sklearn.model_selection import cross_val_score

from models import *


def save_model(model, name):
    with open(name, 'wb') as fid:
        pickle.dump(model, fid)


def read_model(name):
    with open(name, 'rb') as fid:
        model = pickle.load(fid)
    return model


def find_best_model(val_x, val_y):
    # ----------------------- Evaluate Models -----------------
    #xgboost(X_val, y_val)
    #svm(X_val, y_val)
    #rand_forest(X_val, y_val)
    #knn(X_val, y_val)
    #dec_tree(X_val, y_val)


def train(train_x, train_y, val_x, val_y, test):
    # Define best models (with hyperparameter tuning)
    model1, model2, model3 = find_best_model(val_x, val_y)

    # Ensemble modeling
    model = VotingClassifier(estimators=[(
        'm1', model1), ('m2', model2), ('m3', model3)], voting='soft', n_jobs=-1)
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
