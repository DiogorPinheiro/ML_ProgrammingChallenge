import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score


def outliers_plot(data, data_col):
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


def model_result(model, data_x, data_y, best_param):
    cv = StratifiedKFold(n_splits=5, random_state=1)
    nested_score = cross_val_score(model, X=data_x, y=data_y, cv=cv)
    print("Best: %f using %s" % (nested_score.mean(), best_param))


def plot_correlation(data):
    corr = data.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, annot=True)
    plt.show()


def plot_distribution(data):
    col = data.columns
    for c in col:
        try:
            g = sns.distplot(data[c], color="m",
                             label="Skewness : %.2f" % (data[c].skew()))
            g = g.legend(loc="best")
            plt.show()
        except:
            pass


def compare_models(results, std):
    g = sns.barplot("CrossValMeans", "Classifier", data=results,
                    palette="Set2", orient="h", **{'xerr': std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    plt.show()

    # ------------------------ Aux Functions ------------------


def save_model(model, name):
    with open(name, 'wb') as fid:
        pickle.dump(model, fid)


def read_model(name):
    with open(name, 'rb') as fid:
        model = pickle.load(fid)
    return model
