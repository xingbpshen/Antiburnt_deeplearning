import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from DataSetHelper import Difference

usecols = [1, 2]
pred_fut_sec = 15
group_sec = 10
pred_fut_group = 10

len_use_diff = 2
use_diff = False


def normalize(path):
    df = pandas.read_csv(path, skiprows=2, usecols=usecols)
    ds = df.values
    ds = ds.astype('float32')
    ds = np.array(ds)

    # ds[:, 0] = (ds[:, 0] - 2601) / (3427 - 2601)
    # ds[:, 1] = (ds[:, 1] - 4525) / (7740 - 4525)

    scalar = MinMaxScaler(feature_range=(0, 1))
    ds = scalar.fit_transform(ds)
    return np.array(ds)


def create_X_Y_window(ds, group_sec, pred_fut_sec, pred_fut_group):
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


def get_X_Y(path):
    ds = normalize(path)
    X, Y = create_X_Y_window(ds, group_sec, pred_fut_sec, pred_fut_group)
    return np.array(X), np.array(Y)


def get_X_Y_with_diff(path):
    df = pandas.read_csv(path, skiprows=2, usecols=usecols)
    ds = df.values
    ds = ds.astype('float32')
    X, Y = create_X_Y_window(ds, group_sec, pred_fut_sec, pred_fut_group)
    x = Difference.get_diff_for_LSTM(X)
    y = Difference.get_diff_for_LSTM(Y)
    temp_X = X.shape
    temp_Y = Y.shape
    scalar = MinMaxScaler(feature_range=(0, 1))
    X = np.reshape(X, (temp_X[0], temp_X[1]*temp_X[2]))
    Y = np.reshape(Y, (temp_Y[0], temp_Y[1]*temp_Y[2]))
    X = scalar.fit_transform(X)
    Y = scalar.fit_transform(Y)
    X = np.reshape(X, temp_X)
    Y = np.reshape(Y, temp_Y)
    X = np.concatenate((X, x), axis=2)
    Y = np.concatenate((Y, y), axis=2)
    return np.array(X), np.array(Y)


if use_diff:
    len_cols = len(usecols) + len_use_diff
    trainX_2_1, trainY_2_1 = get_X_Y_with_diff(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn02.csv')
    print(trainX_2_1)
    trainX_4_1, trainY_4_1 = get_X_Y_with_diff(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn04.csv')
    # trainX_5_1, trainY_5_1 = get_X_Y(
    #     '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn05.csv')
else:
    len_cols = len(usecols)
    trainX_2_1, trainY_2_1 = get_X_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn02.csv')
    trainX_4_1, trainY_4_1 = get_X_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn04.csv')
    # trainX_4_2_inferior, trainY_4_2_inferior = get_X_Y(
    #     '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn04_2_inferior.csv')


model = Sequential()
model.add(LSTM(32, dropout=0.1, input_shape=[group_sec + 1, len_cols]))
model.add(RepeatVector(pred_fut_group))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(len(usecols))))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(len_cols))

model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])

print('Training...')
model.fit(trainX_2_1, trainY_2_1, epochs=50)
model.fit(trainX_4_1, trainY_4_1, epochs=50)
# model.fit(trainX_4_2_inferior, trainY_4_2_inferior, epochs=50)

joblib.dump(model, '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/saved_models/predictor_1.pkl')
print("Complete, model saved as predictor_1.pkl")

exit()
