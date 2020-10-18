import pickle
from sklearn.model_selection import cross_val_score

from models import *


def save_model(model):
    with open('best_model.pkl', 'wb') as fid:
        pickle.dump(model, fid)


def read_model():
    with open('best_model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    return model


def train(train_x, train_y, val_x, val_y, test):
    # define best model (with hyperparameters)
    model = xgboost(val_x, val_y)
    #model = adaboost(val_x, val_y)
    # Save Model
    save_model(model)

    # Fit training data
    best_model = model.fit(train_x, train_y)

    # Evaluate cross_val_score
    accuracy = cross_val_score(
        model, train_x, train_y, scoring='accuracy', cv=10)
    print("Accuracy of best model in training : {}".format(accuracy.mean() * 100))

    # Predict and return output classes
    output_classes = best_model.predict(test)
    return output_classes
