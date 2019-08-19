import numpy as np


def get_trainX_trainY(window, grouped_X_len, grouped_Y_len, fin_X_shape, fin_Y_shape):
    trainX = []
    trainY = []
    for i in range(0, len(window)):
        x = []
        for jx in range(0, grouped_X_len):
            for kx in window[i][jx]:
                x.append(kx)
        y = []
        for jy in range(grouped_X_len, grouped_X_len + grouped_Y_len):
            for ky in window[i][jy]:
                y.append(ky)
        trainX.append(x)
        trainY.append(y)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    trainX = np.reshape(trainX, (len(window), fin_X_shape[0], fin_X_shape[1]))
    trainY = np.reshape(trainY, (len(window), fin_Y_shape[0], fin_Y_shape[1]))
    return np.array(trainX), np.array(trainY)

