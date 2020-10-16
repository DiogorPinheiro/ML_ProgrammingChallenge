import seaborn as sns
import matplotlib.pyplot as plt


def outliers_detection(data):
    cols = data.columns  # Columns of data

    for c in cols:
        sns.boxplot(x=data[c])
        plt.show()


def contains_null(data):
    cols = data.columns  # Columns of data
    out = []
    null_indexes = []
    for c in cols:
        res = data[c].isnull().values.any()
        out.append(res)
        if res == True:
            null_indexes.append(c)
    return out, null_indexes


def contains_nan(data):
    cols = data.columns  # Columns of data
    out = []
    nan_indexes = []
    for c in cols:
        res = data[c].isna().values.any()
        out.append(res)
        if res == True:
            nan_indexes.append(c)
    return out, nan_indexes
