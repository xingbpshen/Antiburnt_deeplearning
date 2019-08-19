import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.externals import joblib
from version_2.Helper import DataSetHelper, WindowHelper, TrainingSetHelper

"""Change the parameters here
"""
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


trainX, trainY = get([
                      'IR01-1', 'IR01-2', 'IR01-3', 'IR01-4',
                      'IR02-1', 'IR02-2', 'IR02-3', 'IR02-4',
                      'IR03-1', 'IR03-2', 'IR03-3', 'IR03-4',
                      'IR04-1', 'IR04-2', 'IR04-3', 'IR04-4'
                     ],
                     skiprows=skiprows, usecols=usecols, grouped_x_len=grouped_x_len, grouped_y_len=grouped_y_len)

model = Sequential()
model.add(LSTM(40, dropout=0.1, input_shape=(len(usecols), grouped_x_len)))
model.add(Dense(grouped_y_len))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.fit(trainX, trainY, epochs=100, verbose=2)

joblib.dump(model, '/Users/AntonioShen/predictor.pkl')
print("Complete, model saved as predictor.pkl")

exit()
