from sklearn.externals import joblib
from DataSetHelper import X_Y, Difference
import numpy as np

model = joblib.load('/Users/antoniofawcett/PycharmProjects/AntiburntSensor/saved_models/classifier_1.pkl')
path = '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn05_3.csv'
# testX, testY = X_Y.get_X_Y_classification(
#     '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn04_2_inferior.csv')

X, testY = X_Y.get_X_Y_window_for_diff_classification(path)
X = Difference.get_diff(X)
testX, nottestY = X_Y.get_X_Y_classification(path)
testX = np.concatenate((testX, X), axis=1)
score = model.evaluate(testX, testY)
print("Score is: %.2f" % (score[1]*100))

exit()
