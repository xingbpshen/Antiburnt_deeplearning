import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_diff(arr):
    arr = np.reshape(arr, (arr.shape[0], arr.shape[1], arr.shape[2]))
    diff_arr = []
    diff_arr = np.array(diff_arr)
    for i in range(0, len(arr)):
        a = []
        a = np.array(a)
        a = np.append(a, 0)
        a = np.append(a, 0)
        for j in range(0, (len(arr[0]) - 1)):
            t = arr[i][j][0] - arr[i][j + 1][0]
            h = arr[i][j][1] - arr[i][j + 1][1]
            a = np.append(a, t)
            a = np.append(a, h)
        diff_arr = np.append(diff_arr, a)
    # diff_arr = np.reshape(diff_arr, (arr.shape[0], arr.shape[1]*arr.shape[2]))
    # scalar = MinMaxScaler(feature_range=(0, 1))
    # diff_arr = scalar.fit_transform(diff_arr)
    diff_arr = np.reshape(diff_arr, (arr.shape[0], arr.shape[2], arr.shape[1]))
    return np.array(diff_arr)


def get_diff_for_LSTM(arr):
    diff_arr = []
    diff_arr = np.array(diff_arr)
    for i in range(0, len(arr)):
        a = []
        a = np.array(a)
        a = np.append(a, 0)
        a = np.append(a, 0)
        for j in range(0, (len(arr[0]) - 1)):
            t = arr[i][j][0] - arr[i][j + 1][0]
            h = arr[i][j][1] - arr[i][j + 1][1]
            a = np.append(a, t)
            a = np.append(a, h)
        diff_arr = np.append(diff_arr, a)
    # diff_arr = np.reshape(diff_arr, (arr.shape[0], arr.shape[1] * arr.shape[2]))
    # scalar = MinMaxScaler(feature_range=(0, 1))
    # diff_arr = scalar.fit_transform(diff_arr)
    diff_arr = np.reshape(diff_arr, (arr.shape[0], arr.shape[1], arr.shape[2]))
    return np.array(diff_arr)
