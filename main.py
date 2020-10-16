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

from visualization import outliers_detection, contains_null, contains_nan


def data_analysis(data):
    null_out, null_indexes = contains_null(data)

    if null_indexes:    # If contains null, replace it with '?'
        data = data.fillna('?')
    data = data.replace('?', np.NaN)   # Replace null with nan

    nan_out, nan_indexes = contains_nan(data)   # check if contains nan
    if nan_indexes:
        # Replace nan with respective column mean
        data.fillna(data.mean(), inplace=True)

    # outliers_detection(data)    # check if there are outliers

    return data


if __name__ == "__main__":
    # Read csv files
    training = pd.read_csv('data/TrainOnMe.csv', index_col=0, sep=',')
    test = pd.read_csv('data/EvaluateOnMe.csv', index_col=0, sep=',')

    # Split training into labels and data
    train_y = training['y']
    train_x = training.loc[:, training.columns != 'y']
    train_x = data_analysis(train_x)
