import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class IterativeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier=None, rows_percentage=0.34):
        self.base_classifier = base_classifier
        self.rows_percentage = rows_percentage
        self.base_X = None
        self.base_y = None

    def fit(self, X, y):
        self.base_X = X
        self.base_y = y

    def predict(self, X):
        num_of_rows_per_iteration = int(math.ceil(self.rows_percentage * X.shape[0]))
        current_num_of_rows = 0
        current_X = self.base_X
        current_y = self.base_y
        print 'Learning on {} examples'.format(len(current_y))
        self.base_classifier.fit(current_X, current_y)
        while current_num_of_rows < X.shape[0]:
            y_predicted = self.base_classifier.predict(X)
            current_num_of_rows += num_of_rows_per_iteration
            current_X = current_X.append(X.head(current_num_of_rows))
            current_y = np.append(current_y, y_predicted[:current_num_of_rows])
            print 'Learning on {} examples'.format(len(current_y))
            self.base_classifier.fit(current_X, current_y)
        return self.base_classifier.predict(X)
