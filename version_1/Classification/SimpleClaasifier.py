import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
import pandas
from sklearn.externals import joblib
from DataSetHelper import X_Y, Difference

usecols = [1, 2]
pred_fut_sec = 15
group_sec = 10
pred_fut_group = 10
use_external_cols = [8, 9]

use_diff = False
len_use_diff = 2


def normalize(source):
    frame = pandas.read_csv(source, skiprows=2, usecols=usecols)
    y = pandas.read_csv(source, skiprows=2, usecols=use_external_cols)
    ds = frame.values
    train_y = y.values
    ds = ds.astype('float32')
    train_y = train_y.astype('float32')
    ds = np.array(ds)
    train_y = np.array(train_y)

    # ds[:, 0] = (ds[:, 0] - 2601) / (3427 - 2601)
    # ds[:, 1] = (ds[:, 1] - 4525) / (7740 - 4525)

    scalar = MinMaxScaler(feature_range=(0, 1))
    ds = scalar.fit_transform(ds)
    return np.asarray(ds), np.asarray(train_y)


def create_window(dsX, dsY, group_sec, pred_fut_sec, pred_fut_group):
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


def get_X_Y_for_window(dsX, dsY):
    X = np.reshape(dsX, (dsX.shape[0], len(usecols), dsX.shape[1]))
    Y = np.expand_dims(dsY, axis=2)
    return np.array(X), np.array(Y)


def get_X_Y(path):
    ds_x, ds_y = normalize(path)
    trainX, trainY = create_window(ds_x, ds_y, group_sec, pred_fut_sec, pred_fut_group)
    trainX, trainY = get_X_Y_for_window(trainX, trainY)
    return np.array(trainX), np.array(trainY)


def get_X_diff_and_Y(path):
    X, Y = X_Y.get_X_Y_window_for_diff_classification(path)
    X = Difference.get_diff(X)
    trainX, trainY = get_X_Y(path)
    trainX = np.concatenate((trainX, X), axis=1)
    return np.array(trainX), np.array(Y)


if use_diff:
    col_num = len(usecols) + len_use_diff
    trainX_2_1, trainY_2_1 = get_X_diff_and_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn02(modified).csv')
    trainX_4_1, trainY_4_1 = get_X_diff_and_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn04(modified).csv')
    trainX_5_1, trainY_5_1 = get_X_diff_and_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn05.csv')
else:
    col_num = len(usecols)
    trainX_2_1, trainY_2_1 = get_X_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn02(modified).csv')
    trainX_4_1, trainY_4_1 = get_X_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn04(modified).csv')
    trainX_5_1, trainY_5_1 = get_X_Y(
        '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn05.csv')

model = Sequential()
model.add(Dense(32, input_shape=(col_num, group_sec + pred_fut_group + 1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Training...")
model.fit(trainX_2_1, trainY_2_1, epochs=50)
model.fit(trainX_5_1, trainY_5_1, epochs=50)
model.fit(trainX_4_1, trainY_4_1, epochs=50)

joblib.dump(model, '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/saved_models/classifier_1.pkl')
print("Complete, model saved as classifier_1.pkl")

exit()
