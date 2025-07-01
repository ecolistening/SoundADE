import numpy as np
import pandas as pd
import dask

channel1: list[dict] = [
    {'country': 'ecuador', 'habitat': 0, 'recorder': 4},
    {'country': 'ecuador', 'habitat': 1, 'recorder': 9}
]


def by_criteria(df: pd.DataFrame, criteria: pd.DataFrame, on: list, not_in=False) -> pd.DataFrame:
    """
    Filter rows from a DataFrame based on criteria DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to filter.
        criteria (pd.DataFrame): Criteria DataFrame.
        on (list): List of column names to merge on.
        not_in (bool, optional): Flag indicating whether to keep rows not in criteria. Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        >>> criteria = pd.DataFrame({'A': [1], 'B': ['a']})
        >>> by_criteria(df, criteria, on=['A', 'B'])
           A  B
        0  1  a
    """
    merge = df.merge(criteria[on], on=on, how='left', indicator='exists')

    if not_in:
        merge = merge[merge.exists == 'left_only'].drop(columns='exists')
    else:
        merge = merge[merge.exists == 'both'].drop(columns='exists')

    return merge


def channels(df: pd.DataFrame, channel_dicts: list[dict] = channel1) -> pd.DataFrame:
    """
    Filter DataFrame to keep only specific channels.

    Args:
        df (pd.DataFrame): Input DataFrame.
        channel_dicts (list[dict], optional): List of dictionaries representing the channels to keep.
            Each dictionary should contain the keys 'country', 'habitat', and 'recorder' with corresponding values.
            Defaults to channel1, which is defined as [{ 'country': 'ecuador', 'habitat': 0, 'recorder': 4 },
            { 'country': 'ecuador', 'habitat': 1, 'recorder': 9 }].


    Returns:
        pd.DataFrame: Filtered DataFrame.

    Examples:
        >>> df = pd.DataFrame({'channel': [0, 0, 0, 1], 'country': ['ecuador', 'ecuador', 'uk', 'ecuador'],
        ...                    'habitat': [0, 1, 2, 0], 'recorder': [4, 9, 6, 4]})
        >>> channel_dicts = [{'country': 'ecuador', 'habitat': 0, 'recorder': 4}]
        >>> channels(df, channel_dicts)
           channel  country  habitat  recorder
        1        0  ecuador        1         9
        2        0       uk        2         6
        3        1  ecuador        0         4
    """
    ch1 = pd.DataFrame(channel_dicts)

    # Rows where the country/habitat/recorder match those where we want to keep channel 1
    idx_matches_ch1 = df.set_index(['country', 'habitat', 'recorder']).index.isin(
        ch1.set_index(['country', 'habitat', 'recorder']).index)

    # XNOR the channel and those where we want to keep 1 -> ~(0 ^ False) = True, ~(1 ^ True) = True
    df_good = df[np.logical_not(np.logical_xor(df.channel, idx_matches_ch1))]
    return df_good



def first_and_last_days(df: pd.DataFrame, groupby='location', date_column='date', count_column='timestamp',
                        dask=False) -> pd.DataFrame:
    """
    Remove the first and last recorded dates from each group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groupby (str, optional): Column name containing groups from which to remove first and last dates.
        date_column (str, optional): Column name containing dates. Defaults to 'date'.
        count_column (str, optional): Column name containing counts. Defaults to 'timestamp'.
        dask (bool, optional): Flag indicating whether the DataFrame is a Dask DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with first and last days removed.
    """
    first_and_last = df.groupby(groupby)[date_column].aggregate(['min', 'max']).melt()[count_column]

    if dask:
        first_and_last = first_and_last.compute()

    first_and_last = first_and_last.to_list()

    print(f'Removing {df.date.isin(first_and_last).sum()} dates.')

    #TODO HACK This only works because the dates for the different sites don't overlap
    return df[~df[date_column].isin(first_and_last)]


def days_with_too_few_points(df: pd.DataFrame, groupby=['location'], agg_columns=['date'], count_column='timestamp',
                             stds=1.0, dask=False) -> pd.DataFrame:
    """
    Filter DataFrame to remove groups with too few points.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groupby (list, optional): Column names to group by. Defaults to ['location'].
        agg_columns (list, optional): Column names to aggregate. Defaults to ['date'].
        count_column (str, optional): Column name containing counts. Defaults to 'timestamp'.
        stds (float, optional): Number of standard deviations for determining too few points. Defaults to 1.0.
        dask (bool, optional): Flag indicating whether the DataFrame is a Dask DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame.

    Examples:
        >>> df = pd.DataFrame({'location': ['A', 'A', 'A', 'A', 'A', 'A', 'A'],
        ...                    'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02',
        ...                             '2023-01-03', '2023-01-03', '2023-01-04'],
        ...                    'timestamp': [1, 2, 3, 4, 5, 6, 7]})
        >>> days_with_too_few_points(df, groupby=['location'])
          location        date  timestamp
        0        A  2023-01-01          1
        1        A  2023-01-01          2
        2        A  2023-01-02          3
        3        A  2023-01-02          4
        4        A  2023-01-03          5
        5        A  2023-01-03          6
    """
    if dask:
        daily_counts = df.groupby(by=groupby + agg_columns).agg({count_column: 'count'}).reset_index().compute()
    else:
        daily_counts = df.groupby(by=groupby + agg_columns).agg({count_column: 'count'}).reset_index()

    daily_count_loc_mean = daily_counts.groupby(groupby)[count_column].transform('mean')
    daily_count_loc_std = daily_counts.groupby(groupby)[count_column].transform('std')

    # Assume datasets with only one aggregator member (date, recorder) should be kept:
    daily_count_loc_std = daily_count_loc_std.fillna(1.0)

    good_dates = daily_counts[np.abs(daily_counts[count_column] - daily_count_loc_mean) / daily_count_loc_std < stds]

    return by_criteria(df, good_dates, on=groupby + agg_columns)


def recorders_with_too_few_points(df: pd.DataFrame, groupby=['location'], agg_columns=['date', 'recorder'],
                                  count_column='timestamp', dask=False) -> pd.DataFrame:
    """
    Filter DataFrame to remove groups with recorders having too few points.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groupby (list, optional): Column names to group by. Defaults to ['location'].
        agg_columns (list, optional): Column names to aggregate. Defaults to ['date', 'recorder'].
        count_column (str, optional): Column name containing counts. Defaults to 'timestamp'.
        dask (bool, optional): Flag indicating whether the DataFrame is a Dask DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    daily_rec_counts = df.groupby(by=groupby + agg_columns).agg({count_column: 'count'}).reset_index()
    daily_rec_count_loc_mean = daily_rec_counts.groupby(groupby)[count_column].transform('mean')
    daily_rec_count_loc_std = daily_rec_counts.groupby(groupby)[count_column].transform('std')

    date_rec_match = df.set_index(groupby + agg_columns).index.isin(daily_rec_counts[np.abs(
        daily_rec_counts[count_column] - daily_rec_count_loc_mean) / daily_rec_count_loc_std < 1.0].set_index(
        groupby + agg_columns).index)

    return df[date_rec_match]


def first_n_days(df: dask.dataframe.DataFrame, groupby='location', n=10, date_column='date', dask=False) -> pd.DataFrame:
    """
    Keep only the first n days worth of data for each group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groupby (str, optional): Column name to group by. Defaults to 'location'.
        n (int, optional): Number of days to keep. Defaults to 10.
        date_column (str, optional): Column name containing dates. Defaults to 'date'.
        dask (bool, optional): Flag indicating whether the DataFrame is a Dask DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame.

    Examples:
        >>> df = pd.DataFrame({'location': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        ...                    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
        ...                             '2023-02-01', '2023-02-02', '2023-02-03', '2023-02-04'],
        ...                    'timestamp': [1, 2, 3, 4, 5, 6, 7, 8]})
        >>> df.date = pd.to_datetime(df.date)
        >>> first_n_days(df, groupby='location', n=2)
          location       date  timestamp
        0        A 2023-01-01          1
        1        A 2023-01-02          2
        4        B 2023-02-01          5
        5        B 2023-02-02          6
    """
    if dask:
        dates = df[['location', 'date']].drop_duplicates(keep='first').sort_values('date').groupby("location").head(n=n).reset_index(drop=True).compute()

        return by_criteria(df, dates, on=['location', 'date'])

    else:
        location_dates = df[~df.duplicated(subset=[groupby, date_column], keep='first')]. \
            groupby(groupby)[date_column].nsmallest(n).reset_index().drop(columns='level_1')

        # Rows where the country/habitat/recorder match those where we want to keep channel 1
        locdate_match = df.set_index(['location', 'date']).index.isin(
            location_dates.set_index(['location', 'date']).index)

        # XNOR the channel and those where we want to keep 1 -> ~(0 ^ False) = True, ~(1 ^ True) = True
        df_good = df[locdate_match]
        return df_good

    #TODO ChatGPT Code. Check if this works. If so replace my code.
    # if dask:
    #     return df.groupby(groupby).apply(lambda x: x.nsmallest(n, date_column)).reset_index(drop=True).compute()
    # else:
    #     return df.groupby(groupby).apply(lambda x: x.nsmallest(n, date_column)).reset_index(drop=True)
