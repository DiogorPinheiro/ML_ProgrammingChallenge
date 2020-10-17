'''
    ------------------------------------------  Programming Challenge -------------------------------------------------------------

                                        KTH Royal Institute of Technology
                                            M.Sc Machine Learning 20/21

                                        DD2421- Machine Learning

                                                Diogo Pinheiro

    -------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from visualization import outliers_detection, contains_null, contains_nan
from models import *


def replace_outliers(data):
    for c in data.columns:
        if (c != 'x5') and (c != 'x6'):
            # Quantile-based Flooring and Capping
            q_min = data[c].quantile(0.05)
            q_max = data[c].quantile(0.95)

            # Replace by median
            data[c] = np.where(data[c] > q_max, q_min, data[c])
    return data


def data_analysis(data):
    data_col = data.columns
    data.replace(r'\s+( +\.)|#', np.nan, regex=True).replace('', np.nan)
    data.replace('?', np.nan, inplace=True)

    # null_out, null_indexes = contains_null(data, data_col) # check if there's any empty cells
    # nan_out, nan_indexes = contains_nan(data, data_col)   # check if contains nan
    data.fillna(data.mean(), inplace=True)

    data.to_csv('out.csv', index=False)
    # print(data.isna().sum())  # Check how many nan values each column has

    # outliers_detection(data, data_col)    # check if there are outliers
    # data = replace_outliers(data)  # Replace outliers if it makes sense

    return data


if __name__ == "__main__":
    # -------------------------------- Read files ----------------------------------
    training = pd.read_csv('data/TrainOnMe.csv',
                           index_col=0, sep=',', na_values=["?"])
    test = pd.read_csv('data/EvaluateOnMe.csv', index_col=0,
                       sep=',', na_values=["?"])

    train_y = training['y']
    train_x = training.loc[:, training.columns != 'y']
    features = list(train_x.columns)
    train_x = data_analysis(train_x)
    # --------------------------------- Encoder -------------------------------------

    # Encode target labels
    label_enc = LabelEncoder()
    transformed_target = label_enc.fit_transform(
        train_y)  # Atsuto = 0 ; Bob = 1 ; Jörg = 2

    # Cat Boost Encoder (transform categorical features into numerical features)
    CBE_encoder = CatBoostEncoder()
    train_x = CBE_encoder.fit_transform(train_x[features], transformed_target)
    test = CBE_encoder.transform(test[features])

    # ---------------------- Split training into labels and data -------------------

    X_train, X_val, y_train, y_val = train_test_split(train_x, transformed_target,
                                                      test_size=0.2,
                                                      random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --------------------------------- Model Evaluation -----------------------------
    xgboost(X_train, y_train)
    #svm(X_train, y_train)
    #rand_forest(X_train, y_train)
    #knn(X_train, y_train)
    #dec_tree(X_train, y_train)

    # -------------------------------- Testing ----------------------------------------

    # --------------------------------- Output ----------------------------------------
    # Get Classes (in order)
    # Decode Classes
    # Export to file
