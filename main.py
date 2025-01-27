'''
    ------------------------------------------  Programming Challenge -------------------------------------------------------------

                                        KTH Royal Institute of Technology
                                            M.Sc Machine Learning 20/21

                                            DD2421- Machine Learning

                                                Diogo Pinheiro

    -------------------------------------------------------------------------------------------------------------------------
'''

from training import *
from models import *
from visualization import *
import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
import featuretools as ft


def replace_outliers(data):
    # Adapted from https://gist.github.com/joseph-allen/14d72af86689c99e1e225e5771ce1600
    outlier_indices = []
    for c in data.columns:
        try:
            # 1st quartile (25%)
            Q1 = np.percentile(data[c], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(data[c], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = data[(data[c] < Q1 - outlier_step)
                                    | (data[c] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        except:
            pass

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > 1)
    return multiple_outliers


def data_analysis(data):
    data_col = data.columns  # Get data columns

    # ---------------------- Null/NaN Task Force ----------------------------
    data.replace(r'\s+( +\.)|#', np.nan, regex=True).replace('', np.nan)
    data.replace('?', np.nan, inplace=True)

    # null_out, null_indexes = contains_null(data, data_col) # check if there's any empty cells
    # nan_out, nan_indexes = contains_nan(data, data_col)   # check if contains nan
    data.fillna(data.mean(), inplace=True)
    # print(data.isna().sum())  # Check how many nan values each column has

    # -------------------- Skewness Task Force ------------------------------

    # plot_distribution(data)    # Check distribution of each column
    # Transform values with log function to reduce skewness distribution
    data["x7"] = data["x7"].map(lambda i: np.log(i) if i > 0 else 0)
    data["x8"] = data["x8"].map(lambda i: np.log(i) if i > 0 else 0)

    # -------------------- Exploratory Data Analysis ----------------------------

    # plot_correlation(data)  # Plot correlation between data features (heatmap)

    return data


if __name__ == "__main__":
    # -------------------------------- Read files ----------------------------------
    training = pd.read_csv('data/TrainOnMe.csv',
                           index_col=0, sep=',', na_values=["?"])
    # Reset index due to rows being removed manually
    training = training.reset_index(drop=True)

    test = pd.read_csv('data/EvaluateOnMe.csv', index_col=0,
                       sep=',', na_values=["?"])

    train_y = training['y']
    train_x = training.loc[:, training.columns != 'y']
    features = list(train_x.columns)

    # ---------------------------------- Outliers ------------------------------------

    # outliers_plot(train_x, data_col)    # check if there are outliers
    outliers = replace_outliers(train_x)  # Replace outliers if it makes sense

    # Drop rows that, according to the Tukey method, contain more than 1 outlier
    train_x = train_x.drop(outliers, axis=0).reset_index(drop=True)
    train_y = train_y.drop(outliers, axis=0).reset_index(drop=True)
    # train_x.to_csv('new_train.csv')

    # ---------------------------------- Pre-Processing ------------------------------
    len_train = len(train_x)
    dataset = pd.concat(objs=[train_x, test], axis=0).reset_index(
        drop=True)    # Concatenate training and test datasets
    dataset = data_analysis(dataset)
    # ---------------------------------- Feature Engineering -------------------------
    #dataset['x11'] = dataset['x9'] * dataset['x4']
    #dataset['x12'] = dataset['x3'] * dataset['x4']

    # Deep Feature Synthesis

    #es = ft.EntitySet(id='TrainOnMe')
    # es = es.entity_from_dataframe(
    #    entity_id='x', dataframe=dataset, index='index')
    # feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='x',
    #                                      trans_primitives=[
    #                                          'multiply_numeric'],
    #                                      verbose=True)
    #feature_matrix.to_csv('new_train.csv', index=False)

    # ---------------------------------- Separate Dataset ----------------------------
    train_x = dataset[:len_train]
    test = dataset[len_train:]
    # --------------------------------- Encoder --------------------------------------
    features = list(train_x.columns)
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
    test = scaler.transform(test)

    # -------------------------------- Training ----------------------------------------
    output_classes = train(X_train, y_train, X_val, y_val, test)
    # --------------------------------- Output ----------------------------------------

    # Decode Classes
    output_classes = label_enc.inverse_transform(output_classes)
    # print(output_classes)

    # Export to file
    with open('103010.txt', 'w') as f:
        for item in output_classes:
            f.write("%s\n" % item)
