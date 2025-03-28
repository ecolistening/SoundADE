from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dac.data.filter import by_criteria

#TODO CHECK THE PREPROCESSING. We should standardize the data first
def PDLogisticRegression(df: pd.DataFrame, first_col, scale=False, random_state=None):
    train = df.loc[df.set == 'train',:]
    test = df.loc[df.set == 'test',:]

    X_train = train.loc[:, first_col:].to_numpy()
    y_train = train['habitat code'].apply(lambda l: {'UK': 0, 'EC': 3}[l[:2]] + int(l[2]) - 1)
    X_test = test.loc[:, first_col:].to_numpy()
    y_test = test['habitat code'].apply(lambda l: {'UK': 0, 'EC': 3}[l[:2]] + int(l[2]) - 1)

    #TODO It needs to be one recorder in each habitat, right? Renumber recorders for this.
    # groups = df_train['recorder']
    labels = train['habitat code']
    name = ''#f'{df_train.country.iloc[0]}'

    return LogisticRegressionLeaveOneGroupOutCV(X_train, y_train, X_test, y_test, labels, name, scale,
                                                random_state=random_state)

def LogisticRegressionLeaveOneGroupOutCV(X_train, y_train, X_test, y_test,
                                         labels, name, scale=False, random_state=None):

    if scale:
        estimator = Pipeline([('scaler', preprocessing.RobustScaler()),
                              ('lr', LogisticRegressionCV(penalty='l1',
                                                          solver='liblinear',
                                                          random_state=random_state
                                          ))]).fit(X_train, y_train)
    else:
        estimator = Pipeline([('lr', LogisticRegressionCV(penalty='l1',
                                                          solver='liblinear',
                                                          random_state=random_state
                                          ))]).fit(X_train, y_train)

    y_predict = estimator.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)

    return pd.Series({
        'estimator': estimator,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'predicted': y_predict,
        'accuracy': accuracy,
        'labels': labels,
        'name': name
    })

def logistic_regression_accuracy_range(df, first_col, set, n=100, by=['country']):
    '''
    Run logistic regression n times return the accuracy for each run.

    :param df: pd.DataFrame, the data to run the logistic regression on
    :param first_col: str, the first column to include in the logistic regression
    :param set: str, the name of the set to run the logistic regression on, stored in the resulting DataFrame
    :param n: int, the number of times to run the logistic regression
    :param by: list, the columns to group by before running the logistic regression
    :return: pd.DataFrame, the accuracy for each run, with the set name and seed number
    '''
    accuracies = None

    for i in range(n):
        _logreg_all_feature_astro = df.reset_index().groupby(by=by).apply(PDLogisticRegression, first_col=first_col, random_state=i)

        accuracy = _logreg_all_feature_astro.accuracy.to_frame()
        accuracy['set'] = set
        accuracy['seed'] = i

        if accuracies is None:
            accuracies = accuracy
        else:
            accuracies = pd.concat([accuracies, accuracy], axis=0)

    return accuracies

def split_combine(df, train, test):
    df_train = by_criteria(df, train, on=['habitat code','recorder','date'])
    df_test = by_criteria(df, test, on=['habitat code','recorder','date'])
    df_tt = pd.concat([df_train.assign(set='train'), df_test.assign(set='test')]).reindex(columns=['set'] + df_train.columns.tolist())
    return df_tt

def plot_confusion(df_country: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(10,10))
    plt.suptitle(f'{df_country.name}')
    for (i, r), ax in zip(df_country.sort_values(by='accuracy',ascending=False).iterrows(), axes.flatten()):
        cmd = ConfusionMatrixDisplay.from_predictions(r.y_test.to_numpy(), r.predicted, ax=ax, colorbar=False, #display_labels=[k for k,v in location_codes.items() if v in r.predicted],
                                                      normalize='true')#, im_kw={'vmin':0.0,'vmax':1.0})
        ax.title.set_text(f'{r.feature} ({r["accuracy"]:.2f})')
        ax.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    cax.grid(False)
    cbar = fig.colorbar(cmd.im_, cax=cax)
    
    return None

def plot_multifeature_confusion(df: pd.DataFrame, figsize=(10,5)):
    '''Plot the confusion matrix for each country in the DataFrame.
    
    :param df: pd.DataFrame, dataframe with the results of the logistic regression
    :param figsize: tuple, the size of the figure
    :return:
    '''
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for (i, r), ax in zip(df.iterrows(), axes.flatten()):
        cmd = ConfusionMatrixDisplay.from_predictions(r.y_test.to_numpy(), r.predicted, ax=ax, colorbar=False, normalize='true')
        ax.title.set_text(f'{r.country} ({r["accuracy"]:.2f})')
        ax.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    cax.grid(False)
    cbar = fig.colorbar(cmd.im_, cax=cax)

    return None

def representative_example(df: pd.DataFrame, first_col: str,
                           accuracies: pd.DataFrame, by=['country'], measure='median'):
    accuracy_marginal_median = accuracies.reset_index().groupby(by=by).agg({'accuracy': measure}).unstack()
    accuracy_country_vector = accuracies.reset_index().set_index(['set', 'seed', 'country']).unstack()

    idx = np.linalg.norm(accuracy_country_vector - accuracy_marginal_median, axis=1).argmin()

    return df.reset_index().groupby(by=by).apply(PDLogisticRegression, first_col=first_col, random_state=idx)
