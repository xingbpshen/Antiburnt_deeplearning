import numpy as np
from sklearn.externals import joblib

path = ''
model_pred = joblib.load(path)
model_classif = joblib.load(path)


def antiburnPredict(list_input):
    flag = int(0)
    input = np.array(list_input)
    input = input.astype('float32')

    for i in range(0, len(input)):
        if input[i][0] == -1:
            return flag

    input = np.delete(input, 0, axis=1)
    input = np.reshape(input, (11, 2))
    input[:, 0] = (input[:, 0] - 2601) / (3427 - 2601)
    input[:, 1] = (input[:, 1] - 4717) / (6485 - 4717)
    input = np.reshape(input, (1, 11, 2))
    pred = model_pred.predict(input)
    a = np.append(input, pred)
    a = np.reshape(a, (1, 2, 21))
    classi = model_classif.predict(a)

    if classi[0][1] / (classi[0][1] + classi[0][0]) > 0.80:
        flag = 1
    else:
        flag = 0

    return int(flag)
