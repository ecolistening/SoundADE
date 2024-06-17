import pandas as pd

<<<<<<< Updated upstream
from soundade.data import by_criteria
=======
from soundade.data.filter import by_criteria
>>>>>>> Stashed changes


def whole_recorders(df, n=2):
    '''Select whole recorders (per location) to hold out for the testing set

    :param df:
    :param n:
    :param exclude:
    :return:
    '''

    testing_location_recorders = df.drop_duplicates(subset=['habitat code', 'recorder'])
    testing_location_recorders = testing_location_recorders.groupby('habitat code').sample(n=n)

    return testing_location_recorders


def counts(df, testing_location_recorders: pd.DataFrame):
    n_total = df.shape[0]
    # n_test_recorder = df.set_index(['location', 'recorder']).index.isin(testing_location_recorders.index).sum()
    n_test_recorder = by_criteria(df, testing_location_recorders, on=['habitat code', 'recorder']).shape[0]

    # df_training_set = df[~df.set_index(['location', 'recorder']).index.isin(testing_location_recorders.index)]
    df_training_set = by_criteria(df, testing_location_recorders, on=['habitat code', 'recorder'], not_in=True)

    n_recorder_days_remaining = df_training_set.drop_duplicates(subset=['habitat code', 'recorder', 'date']).shape[0]
    n_test_recorder_days = n_recorder_days_remaining * (0.3 - n_test_recorder / n_total) / (
            1.0 - n_test_recorder / n_total)
    n_test_recorder_days_per_location = int(n_test_recorder_days // 6)

    return df_training_set, n_test_recorder_days_per_location


def testing_location_recorder_dates(df_training_set, n):
    testing_lrd = df_training_set.drop_duplicates(subset=['habitat code', 'recorder', 'date'])
    testing_lrd = testing_lrd.groupby('habitat code').sample(n=n)

    return testing_lrd


def train_test_split(df, n_recorders=2):
    '''Generates a dataframe of habitat code/recorder/date for a training and testing set.
    Data can be filtered using the `by_criteria` function and this dataframe.

    :param df:
    :param n_recorders:
    :return:
    '''
    testing_location_recorders = whole_recorders(df, n=n_recorders)

    df_training_set, n_test_recorder_days_per_location = counts(df, testing_location_recorders)

    testing_lrd = testing_location_recorder_dates(df_training_set, n_test_recorder_days_per_location)

    # df_lr = df.set_index(['location', 'recorder'])

    in_location_recorder = by_criteria(df, testing_location_recorders, on=['habitat code', 'recorder'])
    in_location_recorder_dates = by_criteria(df, testing_lrd, on=['habitat code', 'recorder', 'date'])

    # df_test = df[np.logical_or(df_lr.index.isin(testing_location_recorders.index),
    #                            df_lr.set_index('date', append=True).index.isin(
    #                                testing_lrd.index))].set_index(['location', 'recorder', 'date'])
    df_test = pd.concat([in_location_recorder, in_location_recorder_dates])

    # Exclude df_test from train set
    df_train = by_criteria(df, df_test, on=['habitat code', 'recorder', 'date'], not_in=True)

    return df_train[['habitat code','recorder','date']].drop_duplicates().reset_index(drop=True), df_test[['habitat code','recorder','date']].drop_duplicates().reset_index(drop=True)