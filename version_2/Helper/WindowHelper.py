import numpy as np


def get_window_2D(ds, window_size):
    """
    Get version_2 2-dimensional continuous window for the whole data set.
    :param ds: input source data set
    :param window_size: version_2 shape (2, ) array represents size of window
    :return: an array of windows is returned
    """
    rows = window_size[0]
    cols = window_size[1]
    a = []
    for i in range(0, len(ds) - rows + 1):
        b = []
        for j in range(i, i + rows):
            c = []
            for k in ds[j]:
                c.append(k)
            b.append(c)
        a.append(b)
    a = np.array(a)
    a = np.reshape(a, (len(ds) - rows + 1, rows, cols))
    return np.array(a)


def get_last_label_1d(ds, group_num):
    l = []
    for i in range(group_num - 1, len(ds)):
        l.append(ds[i])
    l = np.array(l)
    l = np.reshape(l, (len(l), 1, 1))
    return np.array(l)
