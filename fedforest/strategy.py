from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
import numpy as np


class FedForest:

    def __init__(self, estimators, x_test, y_test, n=100):
        """
        A class used to generate a federated forest

        Attributes
        ----------
        estimators: list or array
            the list of estimators from which the federated forest will be generated
        x_test: array
            a sample dataset with which the estimators will be evaluated
        y_test: array
            the label of the sample dataset
        n: int
            the number of estimators to be selected for the FedForest (default=100)
        """
        self.estimators = estimators
        self.x_test = x_test
        self.y_test = y_test
        self.n = n

        if self.n < 10:
            raise ValueError("n must be greater than 10")

    def random_select(self, num_forest):
        """
        Generates a number of forest from the estimators randomly, all having n estimators. The forest are then
        evaluated and the best forest is selected

        Parameters
        ----------
        num_forest: int
            The number of random forests to be generated

        :return: The best performing forest as a random forest classifier
        """

        leading_est = RandomForestClassifier(n_estimators=100).fit(self.x_test, self.y_test)
        test_est = RandomForestClassifier(n_estimators=100).fit(self.x_test, self.y_test)

        test = self.n / 2 if self.n % 2 == 0 else self.n / 2 + 1
        rs = ShuffleSplit(n_splits=num_forest, train_size=self.n / 2, test_size=test, random_state=101)
        indices = []
        for train_index, test_index in rs.split(self.estimators):
            indices.append(np.array([*train_index, *test_index]))

        for i in range(len(indices)):
            if i == 0:
                leading_est.estimators_ = self.estimators[indices[i]]
            test_est.estimators_ = self.estimators[indices[i]]
            if np.sum(leading_est.predict(self.x_test)) <= np.sum(test_est.predict(self.x_test)):
                leading_est.estimators_ = self.estimators[indices[i]]

        return leading_est

    def select(self):
        pass

    def fit(self, method="acc or bic", lower=True):
        pass
