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
