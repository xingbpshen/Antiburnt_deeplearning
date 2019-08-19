import numpy as np
from sklearn.externals import joblib


class AntiburnModules:
    """case: indicates the case number, 0 is IR test case1, 1 is IR test case2, 2 is IR test case4, 3 is IR test case5
    """
    case = -1

    def __init__(self, path):
        self.model_case_classification = joblib.load(path + '/case_classifier_v2.pkl')
        self.model_prediction = joblib.load(path + '/predictor.pkl')
        self.model_classification_lid = joblib.load(path + '/classifier_lid_beta_v2.pkl')
        self.model_classification_nolid = joblib.load(path + '/classifier_nolid_beta_v2.pkl')
        self.model_classification = self.model_classification_lid

    def antiburnPredict(self, list_input):
        """
        Entrance of the algorithm
        :param list_input: a list contains 60*3 elements
        :return: an integer number indicating advanced warning, 0 represents safe; 1 represents warning
        all (x - 379) / 899 means normalization; all x * 899 + 379 means transform to original value
        """
        input = np.array(list_input)
        input = input.astype('float32')
        input = np.delete(input, 0, axis=1)
        input = np.delete(input, 0, axis=1)
        input = np.reshape(input, (1, 1, 60))
        pred = self.model_prediction.predict((input - 379) / 899)
        pred = np.reshape(pred, (1, 10))
        classifier_input = np.concatenate((np.reshape(input, (1, 60)), (pred * 899 + 379)), axis=1)
        if self.case == -1:
            self.case = self.chooseCase(np.reshape((input - 379) / 899, (1, 60, 1)))
            if self.case == 1:
                self.model_classification = self.model_classification_nolid
        result = self.model_classification.predict(classifier_input)
        if result > 0.5:
            return 1
        else:
            return 0

    def chooseCase(self, once_input):
        """
        A function to choose case number
        :param input: input of (1, 60, 1) shape numpy array
        :return: case number 0-3
        """
        temp = self.model_case_classification.predict(once_input)
        return np.argmax(temp, axis=1)
