"""
A collection of utility functions for binning and visualizing data.

Functions:
----------
bin(r, column, upper=0.95, lower=0.05, nbins=10) -> np.ndarray:
    Generate bin edges for a given column of data in a pandas DataFrame.

cut(r, column, bins) -> pd.Series:
    Bin values in a column of data in a pandas DataFrame and count them.

bin_and_cut(r, column, groupby=['country', 'location', 'dddn'], upper=0.95, lower=0.05, nbins=10, reset_index=False) -> pd.DataFrame:
    Apply the bin and cut functions to multiple groups in a pandas DataFrame.

time_of_day_heatmap(drop='feature', index='dddn', *args, **kwargs) -> None:
    Create a heatmap of data grouped by time of day.

"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def bin(r: pd.DataFrame, column: str, upper: float = 0.95, lower: float = 0.05, nbins: int = 10) -> np.ndarray:
    """
    Compute the bin edges based on even spacing between upper and lower quantiles of the entire dataframe.

    Parameters
    ----------
    r : pd.DataFrame
        Input data to compute the bins for.
    column : str
        Column name in the input data frame to compute the bins for.
    upper : float, optional
        The upper quantile to consider for binning (default is 0.95).
    lower : float, optional
        The lower quantile to consider for binning (default is 0.05).
    nbins : int, optional
        The number of bins to compute (default is 10).

    Returns
    -------
    np.ndarray
        The computed bin edges.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> bin(data, 'col', upper=0.9, lower=0.1, nbins=3)
    array([-inf,  2.6,  5.2,  inf])
    >>> bin(data, 'col', upper=0.8, lower=0.2, nbins=4)
    array([-inf,  3.4,  5.8,  8.2,  inf])
    """
    bins = np.linspace(*r[column].quantile([lower, upper]), nbins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    return bins


def cut(r: pd.DataFrame, column: str, bins: np.ndarray) -> pd.DataFrame:
    """
    Cut the data into bins based on the computed bin edges.

    Parameters
    ----------
    r : pd.DataFrame
        Input data to cut into bins.
    column : str
        Column name in the input data frame to cut.
    bins : np.ndarray
        The bin edges computed using the `bin` function.

    Returns
    -------
    pd.DataFrame
        The data binned into the specified bins.
    """
    r = pd.cut(x=r[column], bins=bins).value_counts().sort_index().reset_index(drop=True)
    r.index.rename('bin')
    return r


def bin_and_cut(r: pd.DataFrame, column: str, groupby: List[str] = ['country', 'location', 'dddn'],
                upper: float = 0.95, lower: float = 0.05, nbins: int = 10, reset_index: bool = False) -> pd.DataFrame:
    """
    Compute the bins and cut the data into the bins based on the computed bin edges.

    Parameters
    ----------
    r : pd.DataFrame
        Input data to compute the bins and cut into bins.
    column : str
        Column name in the input data frame to compute the bins and cut into bins.
    groupby : List[str], optional
        The columns to group the input data by before computing the bins (default is ['country','location','dddn']).
    upper : float, optional
        The upper quantile to consider for binning (default is 0.95).
    lower : float, optional
        The lower quantile to consider for binning (default is 0.05).
    nbins : int, optional
        The number of bins to compute (default is 10).
    reset_index : bool, optional
        Whether to reset the index of the resulting data frame (default is False).

    Returns
    -------
    pd.DataFrame
        The data binned into the specified bins.
    """
    try:
        # Compute the bins based on the upper and lower values of the entire (ungrouped) dataframe.
        bins = bin(r, column, upper=upper, lower=lower, nbins=nbins)
        binned = r.groupby(by=groupby).apply(cut, column=column, bins=bins)
        binned = binned.div(binned.sum(axis=1), axis=0)
        if reset_index:
            return binned.reset_index()
        return binned
    except ValueError as e:
        print(e)


def time_of_day_heatmap(drop: str = 'feature', index: str = 'dddn', *args, **kwargs) -> None:
    """
    Create a heatmap plot of time series data aggregated by hour of the day.

    Parameters
    ----------
    drop : str, optional
        The column to drop from the plot, by default 'feature'
    index : str, optional
        The column to use as the index, by default 'dddn'
    args, kwargs
        Additional arguments and keyword arguments passed to `sns.heatmap()`.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `data` argument is not provided.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import seaborn as sns
    >>> from matplotlib import pyplot as plt
    >>> from typing import Tuple
    >>>
    >>> def generate_data() -> Tuple[pd.DataFrame, str]:
    ...     index = pd.date_range('2022-01-01', '2022-01-02', freq='30T')
    ...     df = pd.DataFrame({
    ...         'date': index,
    ...         'value': np.random.rand(len(index)),
    ...         'category': np.random.choice(['A', 'B', 'C'], len(index))
    ...     })
    ...     return df, 'date'
    >>>
    >>> data, index = generate_data()
    >>> time_of_day_heatmap(data=data, index=index, cmap='YlOrRd', annot=True)

    """
    data = kwargs.pop('data', None)
    if data is None:
        raise TypeError("Missing required keyword argument: 'data'")

    heatmap_data = data.drop(columns=drop).set_index(index).mul(100).transpose().iloc[::-1]
    sns.heatmap(heatmap_data, ax=plt.gca(), square=True, cbar=False, *args, **kwargs)
