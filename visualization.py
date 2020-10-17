import seaborn as sns
import matplotlib.pyplot as plt


def outliers_detection(data, data_col):
    # sns.boxplot(x=data['x10'])
    # plt.show()
    for c in data_col:
        try:
            sns.boxplot(x=data[c])
            plt.show()
        except:
            pass


def contains_null(data, data_col):
    out = []
    null_indexes = []
    for c in data_col:
        res = data[c].isnull().values.any()
        out.append(res)
        if res == True:
            null_indexes.append(c)
    return out, null_indexes


def contains_nan(data, data_col):
    out = []
    nan_indexes = []
    for c in data_col:
        res = data[c].isna().values.any()
        out.append(res)
        if res == True:
            nan_indexes.append(c)
    return out, nan_indexes


def model_result(model):
    print("Best: %f using %s" % (model.best_score_, model.best_params_))
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
