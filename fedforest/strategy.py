
class FedForest:
    
    def __init__(self, estimators, x_test, y_test, n):
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
            the number of estimators to be selected for the FedForest
        """
        self.estimators = estimators
        self.x_test = x_test
        self.y_test = y_test
        self.n = n
