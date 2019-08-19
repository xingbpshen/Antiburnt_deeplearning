import numpy as np
from version_2.Helper import DataSetHelper, WindowHelper
from sklearn.externals import joblib
from keras.utils import to_categorical

model_pred = joblib.load('/Users/AntonioShen/predictor.pkl')

skiprows = 119 + 200
usecols_x = [1]
usecols_y = [2]
grouped_x_len = 60
grouped_gen_len = 10


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

            """ Test
            """
            b = b * 899 + 379

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


# Input only one file name here
testX, testY = get(['IR04-5'],
                     skiprows=skiprows, usecols_x=usecols_x, usecols_y=usecols_y,
                     grouped_x_len=grouped_x_len)

input_length = 70
model = joblib.load('/Users/AntonioShen/classifier_nolid_beta_v2.pkl')
score = model.evaluate(testX, testY)
print('\n')
print('Test score is: %.4f' % (score[1]*100))
print('\n')
print('Predicted value | Real value')
print('-------------------------------')
tp_0 = 0
fp_0 = 0
fn_0 = 0
tp_1 = 0
fp_1 = 0
fn_1 = 0
for i in range(0, len(testX)):
    input = testX[i]
    input = np.reshape(input, (1, input_length))
    result = model.predict(input)
    if result > 0.5:
        flag = 1
    else:
        flag = 0
    if flag == 0 and testY[i] == 0:
        tp_0 = tp_0 + 1
    if flag == 1 and testY[i] == 1:
        tp_1 = tp_1 + 1
    if flag == 0 and testY[i] == 1:
        fp_0 = fp_0 + 1
        fn_1 = fn_1 + 1
    if flag == 1 and testY[i] == 0:
        fn_0 = fn_0 + 1
        fp_1 = fp_1 + 1
    print(flag, '              |', testY[i])
    # flag = np.argmax(result, axis=1) Do not use this in binary classification, only use it in multi-class
    # print(flag[0], testY[i])  Do not use this in binary classification, only use it in multi-class
    print('-------------------------------')
P_0 = tp_0 / (tp_0 + fp_0)
R_0 = tp_0 / (tp_0 + fn_0)
P_1 = tp_1 / (tp_1 + fp_1)
R_1 = tp_1 / (tp_1 + fn_1)
marco_P = (P_0 + P_1) / 2
marco_R = (R_0 + R_1) / 2
marco_F1 = 2 * marco_P * marco_R / (marco_P + marco_R)
print('Marco-F1 value is', marco_F1)
exit()
