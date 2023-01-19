from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from .evaluator import extract_n_trees
import numpy as np


class FedForest(RandomForestClassifier):

    def __init__(self, x_test, y_test, n=100):
        RandomForestClassifier.__init__(self)
        """
        A Federated Forest Classifier. It inherits the properties of random forests classifier to eliminate the need 
        for reinventing existing wheels. However the additional methods provides a different strategy for generating the 
        forest from trees from other forests. 

        Attributes
        ----------
        x_test: array
            a sample dataset with which the estimators will be evaluated
        y_test: array
            the label of the sample dataset
        n: int
            the number of estimators to be selected for the FedForest (default=100)

        Methods
        -------
        fit_random(num_forest)
            Creates multiple random forests and evaluate them
        fit_acc(self, estimators)
            Creates a federated forest using accuracy as the evaluation criterion
        fit_bic(self, estimators, num_features, lower=True)
            Creates a federated forest using bayesian information criterion.
        """

        self.x_test = x_test
        self.y_test = y_test
        self.n = n
        self.estimators_ = None

        if self.n < 10:
            raise ValueError("n must be greater than 10")

    def fit_random(self, estimators, num_forest=10):
        """
        Generates a number of forest from the estimators randomly, all having n estimators. The forest are then
        evaluated and the best forest is selected

        Parameters
        ----------
        estimators: list or array
            the list of estimators from which the federated forest will be generated
        num_forest: int
            The number of random forests to be generated (default=10)

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
                leading_est.estimators_ = estimators[indices[i]]
            test_est.estimators_ = estimators[indices[i]]
            if np.sum(leading_est.predict(self.x_test)) <= np.sum(test_est.predict(self.x_test)):
                leading_est.estimators_ = estimators[indices[i]]

        return leading_est

    def fit_acc(self, estimators):
        """
        Evaluates and select the best performing estimators from the provided list of estimators based on accuracy and
        creates a new Federated Forest

        Parameters
        ----------
        estimators: list or array
            the list of estimators from which the federated forest will be generated
        :return:
        """
        self.fit(self.x_test, self.y_test)
        self.estimators_ = extract_n_trees(estimators, self.n, self.x_test, self.y_test, 'acc')
        return self

    def fit_bic(self, estimators, num_features, lower=True):
        """
        Evaluates and select the best performing estimators from the provided list of estimators based on BIC and
        creates a new Federated Forest

        Parameters
        ----------
        estimators: list or array
            the list of estimators from which the federated forest will be generated
        num_features: int
            the number of features in the dataset
        lower: boolean
            specifies if the lower bound of upper bound estimators are selected when using BIC.
        :return: returns a FedForest classifier
        """
        if num_features is None:
            raise ValueError("num_features cannot be None")

        self.fit(self.x_test, self.y_test)
        self.estimators_ = extract_n_trees(estimators, self.n, self.x_test, self.y_test, 'bic',
                                           lower=lower, num_features=num_features)

        return self
