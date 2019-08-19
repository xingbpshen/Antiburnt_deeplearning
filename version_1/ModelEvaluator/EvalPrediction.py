from sklearn.externals import joblib
from DataSetHelper import X_Y

model = joblib.load('/Users/antoniofawcett/PycharmProjects/AntiburntSensor/saved_models/predictor_1.pkl')

testX, testY = X_Y.get_X_Y_prediction(
    '/Users/antoniofawcett/PycharmProjects/AntiburntSensor/7_18_AntiburnTest/antiburn05_3.csv')
score = model.evaluate(testX, testY)
print("Score is: %.2f" % (score[1]*100))

exit()
