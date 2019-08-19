import numpy as np
from version_2.Helper import DataSetHelper, WindowHelper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.externals import joblib
from keras.utils import to_categorical

skiprows = 112
usecols_x = [1]
grouped_x_len = 60


def get(data_list, skiprows, usecols_x, grouped_x_len):
    trainX, trainY = [], []
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    for data_name in data_list:
        path = '/Users/AntonioShen/burn_data/' + data_name + '.csv'
        ds_x = DataSetHelper.get_data_set(path=path, skiprows=skiprows, usecols=usecols_x, use80rows=True)
        win_x = WindowHelper.get_window_2D(ds_x, window_size=[grouped_x_len, len(usecols_x)])
        # y = [int(data_name[3]) - 1] * len(win_x)

        """ Test
        """
        y = [int(int(data_name[3]) / 3)] * len(win_x)

        if len(trainX) is 0 and len(trainY) is 0:
            trainX = np.array(win_x)
            trainY = np.array(y)
        else:
            trainX = np.concatenate((trainX, win_x), axis=0)
            trainY = np.concatenate((trainY, y), axis=0)

    trainY = np.reshape(trainY, (len(trainY), 1))
    trainY = to_categorical(trainY)

    return np.array(trainX), np.array(trainY)


trainX, trainY = get([
                      'IR01-1', 'IR01-2', 'IR01-3',
                      'IR02-1', 'IR02-3', 'IR02-4', 'IR02-5',
                      'IR03-1', 'IR03-2', 'IR03-3', 'IR03-4',
                      'IR04-1', 'IR04-2', 'IR04-3', 'IR04-4'
                      ],
                     skiprows=skiprows, usecols_x=usecols_x, grouped_x_len=grouped_x_len)

model = Sequential()
model.add(Conv1D(128, 2, strides=1, padding='same', activation='relu', kernel_initializer='uniform',
                 input_shape=(grouped_x_len, len(usecols_x))))
model.add(Conv1D(128, 2, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(328, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv1D(328, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

""" Test
"""
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=20, verbose=2)

joblib.dump(model, '/Users/AntonioShen/case_classifier_v2.pkl')
print('Complete, saved as case_classifier_v2.pkl')

exit()

