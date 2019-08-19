import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

usecols = [1, 2]
pred_fut_sec = 15
group_sec = 10
pred_fut_group = 10


def normalize_classification(source):
    frame = pandas.read_csv(source, skiprows=2, usecols=usecols)
    y = pandas.read_csv(source, skiprows=2, usecols=[8, 9])
    ds = frame.values
    train_y = y.values
    ds = ds.astype('float32')
    train_y = train_y.astype('float32')
    ds = np.array(ds)
    # ds[:, 0] = (ds[:, 0] - 2601) / (3427 - 2601)
    # ds[:, 1] = (ds[:, 1] - 4525) / (7740 - 4525)

    scalar = MinMaxScaler(feature_range=(0, 1))
    ds = scalar.fit_transform(ds)
    return np.asarray(ds), np.asarray(train_y)


def create_window_classification(dsX, dsY, group_sec, pred_fut_sec, pred_fut_group):
    dataX, dataY = [], []
    for i in range(group_sec, len(dsX) - pred_fut_group + 1 - pred_fut_sec):
        a = []
        for j in range(i - group_sec, i + 1):
            a.append(dsX[j])
        for k in range(i + pred_fut_sec, i + pred_fut_sec + pred_fut_group):
            a.append(dsX[k])
        dataX.append(a)
        dataY.append(dsY[i + pred_fut_sec])
    return np.array(dataX), np.array(dataY)


def get_X_Y_for_window_classification(dsX, dsY):
    X = np.reshape(dsX, (dsX.shape[0], len(usecols), dsX.shape[1]))
    Y = np.expand_dims(dsY, axis=2)
    return np.array(X), np.array(Y)


def get_X_Y_classification(path):
    ds_x, ds_y = normalize_classification(path)
    trainX, trainY = create_window_classification(ds_x, ds_y, group_sec, pred_fut_sec, pred_fut_group)
    trainX, trainY = get_X_Y_for_window_classification(trainX, trainY)
    return np.array(trainX), np.array(trainY)


def normalize_prediction(path):
    df = pandas.read_csv(path, skiprows=2, usecols=usecols)
    ds = df.values
    ds = ds.astype('float32')
    ds = np.array(ds)
    # ds[:, 0] = (ds[:, 0] - 2601) / (3427 - 2601)
    # ds[:, 1] = (ds[:, 1] - 4525) / (7740 - 4525)

    # scalar = MinMaxScaler(feature_range=(0, 1))
    # ds = scalar.fit_transform(ds)

    ds[:, 0] = (ds[:, 0] - 2601) / (3427 - 2601)
    ds[:, 1] = (ds[:, 1] - 4717) / (6485 - 4717)

    return np.array(ds)


def create_X_Y_window_prediction(ds, group_sec, pred_fut_sec, pred_fut_group):
    X_window, Y_window = [], []
    for i in range(group_sec, len(ds) - pred_fut_group + 1 - pred_fut_sec):
        x, y = [], []
        for j in range(i - group_sec, i + 1):
            x.append(ds[j])
        for k in range(i + pred_fut_sec, i + pred_fut_sec + pred_fut_group):
            y.append(ds[k])
        X_window.append(x)
        Y_window.append(y)
    return np.array(X_window), np.array(Y_window)


def get_X_Y_prediction(path):
    ds = normalize_prediction(path)
    X, Y = create_X_Y_window_prediction(ds, group_sec, pred_fut_sec, pred_fut_group)
    return np.array(X), np.array(Y)


def get_X_Y_window_for_diff_classification(path):
    frame = pandas.read_csv(path, skiprows=2, usecols=usecols)
    Y = pandas.read_csv(path, skiprows=2, usecols=[8, 9, 10, 11])
    X = frame.values
    Y = Y.values
    X = X.astype('float32')
    Y = Y.astype('float32')
    X, Y = create_window_classification(X, Y, group_sec, pred_fut_sec, pred_fut_group)
    X, Y = get_X_Y_for_window_classification(X, Y)
    return np.array(X), np.array(Y)
