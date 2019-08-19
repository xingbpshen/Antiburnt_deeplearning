from sklearn.externals import joblib
from version_2.Helper import DataSetHelper, WindowHelper, TrainingSetHelper
import numpy as np
import pandas as pd

skiprows = 1
usecols = [1]

grouped_x_len = 60
grouped_y_len = 10


def get(data_list, skiprows, usecols, grouped_x_len, grouped_y_len):
    trainX, trainY = [], []
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    for data_name in data_list:
        path = '/Users/AntonioShen/burn_data/' + data_name + '.csv'
        ds = DataSetHelper.get_data_set(path=path, skiprows=skiprows, usecols=usecols)
        win = WindowHelper.get_window_2D(ds, window_size=[grouped_x_len + grouped_y_len, len(usecols)])
        trX, trY = TrainingSetHelper.get_trainX_trainY(
            window=win, grouped_X_len=grouped_x_len, grouped_Y_len=grouped_y_len,
            fin_X_shape=[grouped_x_len, len(usecols)],
            fin_Y_shape=[grouped_y_len, len(usecols)])
        if len(trainX) is 0 and len(trainY) is 0:
            trainX = trX
            trainY = trY
        else:
            trainX = np.concatenate((trainX, trX), axis=0)
            trainY = np.concatenate((trainY, trY), axis=0)

    trainX = np.reshape(trainX, (len(trainX), len(usecols), grouped_x_len))
    trainY = np.reshape(trainY, (len(trainY), grouped_y_len))
    return np.array(trainX), np.array(trainY)


testX, testY = get(['IR04-4'],
                   skiprows=skiprows, usecols=usecols, grouped_x_len=grouped_x_len, grouped_y_len=grouped_y_len)

model = joblib.load('/Users/AntonioShen/predictor.pkl')

pred_record = []

# x * 899 + 379 means transform to original value
for i in range(0, len(testX), grouped_x_len + grouped_y_len):
    pred_record = np.concatenate((pred_record, (np.reshape(testX[i], (grouped_x_len, ))*899+379)), axis=0)
    pred_record = np.concatenate(
        (pred_record, (np.reshape(model.predict(np.expand_dims(testX[i], axis=0))*899+379, (grouped_y_len, )))),
        axis=0)

pred_record = np.reshape(pred_record, (len(pred_record), len(usecols)))

pd_data = pd.DataFrame(pred_record, columns=['temp'])
pd_data.to_csv('/Users/AntonioShen/burn_data/GeneratedData/pred04-4.csv')

exit()
