from sklearn.externals import joblib
import numpy as np
from DataSetHelper import X_Y

usecols = [1, 2]
pred_fut_sec = 15
group_sec = 10
pred_fut_group = 10
len_use_diff = 0

col_num = len(usecols) + len_use_diff

path_1 = '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/test05-2.csv'

model_pred = joblib.load('/Users/antoniofawcett/PycharmProjects/AntiburntSensor/saved_models/predictor_1 1.pkl')
model_classif = joblib.load('/Users/antoniofawcett/PycharmProjects/AntiburntSensor/saved_models/classifier_1 1.pkl')

testX_2_2, testY_2_2 = X_Y.get_X_Y_prediction(path_1)
#testX_2_2_f, testY_2_2_f = X_Y.get_X_Y_classification(path_1)
true_count = 0
ahead = 0

for i in range(0, len(testX_2_2)):
    a = np.expand_dims(testX_2_2[i], axis=0)
    pred = model_pred.predict(a)
    a = np.append(a, pred)
    a = np.reshape(a, (1, 2, 21))
    a = np.reshape(a, (1, col_num, group_sec + 1 + pred_fut_group))

    classi = model_classif.predict(a)

    if classi[0][1] / (classi[0][1] + classi[0][0]) > 0.50:
        flag = 1
    else:
        flag = 0

    print(flag)
    #print('%d, %d' % (flag, testY_2_2_f[i][1]))
    print(classi)
    # if flag == 1 and testY_2_2_f[i][1] == 0:
    #     ahead = ahead + 1
    #     print('-----------------------------------------------')
    #
    # if flag == testY_2_2_f[i][1]:
    #     true_count = true_count + 1

# print('Score: %.2f' % (true_count/len(testX_2_2)*100))
# print('Ahead: %d' % ahead)

exit()
