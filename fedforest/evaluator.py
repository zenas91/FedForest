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


def acc(model, x_test, y_test):
    """
    Computes the accuracy of a single estimator
    :param model: the estimator for which the accuracy is to be computed
    :param x_test: a sample dataset with which to evaluate the estimator
    :param y_test: the label of the sample dataset
    :return: returns the number of correctly classified samples
    """
    result = model.predict(x_test)
    return np.sum(result == y_test)


def extract_n_trees(estimators, num_est, x_test, y_test, method, **kwargs):
    """
    Evaluates all the estimators in the list and returns the specified number of estimators.

    Parameters
    ----------
    :param estimators: list or array
        a list of all the estimators to be evaluated.
    :param num_est: int
         the number of estimators to be return after evaluation.
    :param x_test: array
        the sample dataset with which to evaluate the estimators.
    :param y_test: array
        the labels of the sample dataset
    :param method: string
        the method to be adopted for evaluation. Only two inputs are acceptable, namely 'acc' for accuracy and 'bic' for
        bayesian information criterion.
    :param kwargs:
        when BIC is used as evaluation criterion, additional information are needed.
            'lower': a boolean value to determine if the lowest BIC is selected or the topmost.
            'num_features': the number of features in the training dataset needed to compute the complexity of the
            estimator.
    :return: top 'n' estimators according to the evaluation criteria.
    """

    if method.strip() != "bic" and method.strip() != "acc":
        raise ValueError("method only takes the values 'acc' and 'bic' as argument")
    if method.strip() == "bic" and kwargs['lower'] is None:
        raise ValueError("lower(bool) parameter is required")
    if method.strip() == "bic" and kwargs['num_features'] is None:
        raise ValueError("num_features(int) parameter is required")

    perf = []
    for est in estimators:
        perf.append(acc(est, x_test, y_test)) if method.strip() == "acc" else \
            perf.append(bic(est, x_test, y_test, kwargs['num_features']))

    if method.strip() == "acc":
        ind = np.argpartition(perf, -num_est)[-num_est:]
    else:
        ind = np.argpartition(perf, num_est)[:num_est] if kwargs['lower'] else np.argpartition(perf, -num_est)[-num_est:]

    top_trees = list(np.array(estimators)[ind])

    return top_trees
