import pandas as pd
import numpy as np
import scipy

def mutual_information(df, index=None, drop=None):
    '''The Mutual Information I(X;Y) between two variables X and Y

    >> mutual_information(df_feature, drop=['country','feature'], index='location')

    :param df: The histogram dataframe containing columns as X and rows as Y
    :param index: The column to check mutual information against.
    :param drop: Any non-histogram columns
    :return:
    '''
    if drop is not None:
        df = df.drop(columns=drop)

    if index is not None:
        df = df.set_index(index)

    p_xy = df.to_numpy()
    p_xy /= p_xy.sum()
    px = np.sum(p_xy, axis=1)
    px /= px.sum()
    py = np.sum(p_xy, axis=0)
    py /= py.sum()

    px_py = px.reshape(px.size, -1) * py.reshape(-1,py.size)
    mi = np.nansum(p_xy * np.log(p_xy/px_py))

    return pd.Series({'mi': mi})

def conditional_mutual_information(df: pd.DataFrame, x: str, y: str, z: str, tile=None):
    '''

    :param df: Takes a long-form data frame with columns to be used as x, y, and z in the computation of CMI.
    :param x: Continuous column to be histogrammed.
    :param y: Discrete column.
    :param z: Continuous column to be histogrammed.
    :param tile: Column to tile to disaggregate from histogram binning
    :return:
    '''

    hist2d, _, _ = np.histogram2d(df[x], df[z])

    p_xzs = []
    for location, df in df.groupby(level=0):
        blocks = []
        for dddn, df_tod in df.groupby(level=1):
            blocks.append(df_tod.to_numpy())
        p_xz = scipy.linalg.block_diag(*blocks)
        # p_xz /= p_xz.sum()
        p_xzs.append(p_xz.reshape(p_xz.shape[0], -1, p_xz.shape[1]))

    p_xyz = np.concatenate(p_xzs, axis=1)
    p_xyz /= p_xyz.sum()

    # Create marginal distributions
    p_xz = p_xyz.sum(axis=1)
    p_yz = p_xyz.sum(axis=0)
    p_z = p_xyz.sum(axis=0).sum(axis=0)

    # Tile distributions for computation
    p_z_tile = np.tile(p_z.reshape(1, 1, p_z.shape[0]), (40, 6, 1))
    p_xz_tile = np.tile(p_xz.reshape(p_xz.shape[0], 1, p_xz.shape[1]), (1, 6, 1))
    p_yz_tile = np.tile(p_yz.reshape(1, p_yz.shape[0], p_yz.shape[1]), (40, 1, 1))

    mi = np.nansum(p_xyz * np.log((p_z_tile * p_xyz) / (p_xz_tile * p_yz_tile)), axis=(0, 1)).sum()

    return mi