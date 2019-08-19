import numpy as np
from version_2.Helper import DataSetHelper, WindowHelper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, AveragePooling1D, GlobalAveragePooling1D
from sklearn.externals import joblib
from keras.utils import to_categorical

skiprows = 119 + 200
usecols_x = [1]
usecols_y = [2]
grouped_x_len = 60
grouped_gen_len = 10

model_pred = joblib.load('/Users/AntonioShen/predictor.pkl')


def get(data_list, skiprows, usecols_x, usecols_y, grouped_x_len):
    trainX, trainY = [], []
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    for data_name in data_list:
        path = '/Users/AntonioShen/burn_data/' + data_name + '.csv'
        ds_x = DataSetHelper.get_data_set(path=path, skiprows=skiprows, usecols=usecols_x, normalize=False)
        ds_y = DataSetHelper.get_data_set(path=path, skiprows=skiprows, usecols=usecols_y, normalize=False)
        win_x = WindowHelper.get_window_2D(ds_x, window_size=[grouped_x_len, len(usecols_x)])
        a = []
        a = np.array(a)
        for i in range(0, len(win_x)):
            if len(a) is 0:
                a = win_x[i]
            else:
                a = np.concatenate((a, win_x[i]), axis=0)
            b = model_pred.predict(np.reshape(win_x[i], (1, 1, grouped_x_len)))
            b = b * 899 + 379  # Transform to original value

            a = np.concatenate((a, np.reshape(b, (10, 1))))
        a = np.reshape(a, (len(win_x), grouped_x_len + grouped_gen_len, 1))
        win_y = WindowHelper.get_last_label_1d(ds_y, grouped_x_len + grouped_gen_len)

        if len(trainX) is 0 and len(trainY) is 0:
            trainX = np.array(a)
            trainY = np.array(win_y)
        else:
            trainX = np.concatenate((trainX, a), axis=0)
            trainY = np.concatenate((trainY, win_y), axis=0)

        diff = len(trainX) - len(trainY)
        for i in range(0, diff):
            trainX = np.delete(trainX, len(trainX) - 1, axis=0)

    trainY = np.reshape(trainY, (len(trainY), ))
    # trainY = to_categorical(trainY) Do not use this in binary classification, only use it in multi-class

    """ Test
    """
    trainX = np.reshape(trainX, (len(trainX), grouped_x_len + grouped_gen_len))

    return np.array(trainX), np.array(trainY)


# IR01-IR02 train for classifier_lid_beta_v2; IR03-IR04 train for classifier_nolid_beta_v2
trainX, trainY = get([
                      'IR01-1', 'IR01-2', 'IR01-3', 'IR01-4', 'IR02-1', 'IR02-2', 'IR02-3', 'IR02-4',
                      # 'IR03-1', 'IR03-2', 'IR03-3', 'IR03-4', 'IR03-5', 'IR04-1', 'IR04-2', 'IR04-3', 'IR04-4'
                      ],
                     skiprows=skiprows, usecols_x=usecols_x, usecols_y=usecols_y,
                     grouped_x_len=grouped_x_len)

input_length = 70
model = Sequential()
model.add(Embedding(1400, 328, input_length=70))  # The first param must be much greater than all temperature value
model.add(Conv1D(328, 5, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(328, 10, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling1D(pool_size=10))
model.add(Dropout(0.2))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=3, verbose=2)

joblib.dump(model, '/Users/AntonioShen/classifier_lid_beta_v2.pkl')
print('Complete, saved as classifier_lid_beta_v2.pkl')

exit()
