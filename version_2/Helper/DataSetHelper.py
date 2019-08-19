import pandas
import numpy as np

"""This contains methods which can operate data extraction from .csv files.
"""


def normalization(ds, min, max):
    ds = (ds - min) / (max - min)
    return np.array(ds)


def get_data_frame(path, skiprows, usecols, use80rows=False):
    """
    Extract data from version_2 data frame.
    :param path: directory of .csv file
    :param skiprows: index of the last row to skip
    :param usecols: index of columns to read in
    :return: version_2 data frame is returned
    """
    if use80rows is True:
        df = pandas.read_csv(path, skiprows=skiprows, usecols=usecols, encoding='latin-1', nrows=80)
    else:
        df = pandas.read_csv(path, skiprows=skiprows, usecols=usecols, encoding='latin-1')
    return df


def get_data_set(path, skiprows, usecols, normalize=True, use80rows=False):
    if use80rows is True:
        df = get_data_frame(path, skiprows, usecols, use80rows=True)
    else:
        df = get_data_frame(path, skiprows, usecols)
    ds = df.values
    ds = ds.astype('float32')
    if normalize is True:
        # min and max are the minimum value and maximum value across the dataset
        ds = normalization(ds, min=379, max=1278)
    return np.array(ds)
