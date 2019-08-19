import numpy as np
from version_2.Helper import DataSetHelper, WindowHelper
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


testX, temp = get(['IR04-5'], skiprows=skiprows, usecols_x=usecols_x, grouped_x_len=grouped_x_len)
model = joblib.load('/Users/AntonioShen/case_classifier_v2.pkl')

print(testX*899+379)

for i in range(0, len(testX)):
    input = testX[i]
    input = np.reshape(input, (1, 60, 1))
    result = model.predict(input)
    print(np.argmax(result, axis=1))

exit()
