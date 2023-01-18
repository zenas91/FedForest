from sklearn.metrics import mean_squared_error
from math import log
import numpy as np


def bic(model, x_test, y_test, num_features):
    """
    Computes the BIC of a single estimator
    :param model: the estimator for which the BIC is to be computed
    :param x_test: a sample dataset with which to evaluate the estimator
    :param y_test: the label of the sample dataset
    :param num_features: the number of features in the dataset
    :return: the BIC value for the estimator
    """
    cx = log((model.tree_.node_count / model.get_depth()) * model.get_n_leaves() * num_features)
    mse = mean_squared_error(y_test, model.predict(x_test))
    m_bic = log(mse) + cx * log(model.tree_.n_node_samples[0])
    return m_bic


def accuracy(model, x_test, y_test,):
    """
    Computes the accuracy of a single estimator
    :param model: the estimator for which the accuracy is to be computed
    :param x_test: a sample dataset with which to evaluate the estimator
    :param y_test: the label of the sample dataset
    :return: returns the number of correctly classified samples
    """
    result = model.predict(x_test)
    return np.sum(result == y_test)
