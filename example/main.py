from preprocessing.processing import get_train_data, get_test_data
from sklearn.ensemble import RandomForestClassifier
from fedforest.strategy import FedForest
from fedforest.evaluator import extract_n_trees
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def get_n_estimators(model, n, x_test, y_test, method, lower=True, num_features=18):
    """
    Get n best estimators from a forest.
    :param model: A forest based model. This assumes a scikit learn random forest or extra trees classifier is adopted
    because they both have a .estimators_ function for extracting the trees.
    :param n: the number of trees to be returned
    :param x_test: the sample dataset for evaluating the estimators or trees
    :param y_test: the lable of the sample dataset
    :param method: the method of evaluation. Either accuracy or BIC is adopted for tree evaluation
    :param lower: lower indicates weather the trees with the lowest BIC are selected or otherwise
    :param num_features: the number of features in the dataset.
    :return: n best estimators based on the evaluation method is returned.
    """
    estimators = model.estimators_

    if method.strip() == "acc":
        top_n = extract_n_trees(estimators, n, x_test, y_test, 'acc')
    else:
        top_n = extract_n_trees(estimators, n, x_test, y_test, 'bic', lower=lower, num_features=num_features)

    return top_n


if __name__ == '__main__':
    # Slight Equilibrium Distribution

    # inf g
    x_g, y_g = get_train_data("global")
    xt_g, yt_g = get_test_data("global")
    int_g_rf = RandomForestClassifier(n_estimators=100).fit(x_g, y_g)

    # inf 1
    x_1, y_1 = get_train_data("caida")
    int_1_rf = RandomForestClassifier(n_estimators=100, max_depth=6).fit(x_1, y_1)
    top_35_inf_1 = get_n_estimators(int_1_rf, 35, x_g, y_g, method='acc')

    # inf 2
    x_2, y_2 = get_train_data("dos")
    int_2_rf = RandomForestClassifier(n_estimators=100, max_depth=6).fit(x_2, y_2)
    top_35_inf_2 = get_n_estimators(int_2_rf, 35, x_g, y_g, method='acc')

    # inf 3
    x_3, y_3 = get_train_data("ids")
    int_3_rf = RandomForestClassifier(n_estimators=100, max_depth=6).fit(x_3, y_3)
    top_35_inf_3 = get_n_estimators(int_3_rf, 35, x_g, y_g, method='acc')

    est = top_35_inf_1 + top_35_inf_2 + top_35_inf_3

    fed = FedForest(x_g, y_g, 100).fit_acc(est)

    print("FedForest: ", np.sum(fed.predict(x_g) == y_g))
    print(confusion_matrix(y_g, fed.predict(x_g)))
    print(classification_report(y_g, fed.predict(x_g)))

    print("Central: ", np.sum(int_g_rf.predict(x_g) == y_g))
    print(confusion_matrix(y_g, int_g_rf.predict(x_g)))
    print(classification_report(y_g, int_g_rf.predict(x_g)))
