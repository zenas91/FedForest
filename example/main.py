from preprocessing.processing import get_train_data, get_test_data
from sklearn.ensemble import RandomForestClassifier
from fedforest.strategy import FedForest
from fedforest.evaluator import extract_n_trees
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Slight Equilibrium Distribution

    x, y = get_train_data("caida")
    xt, yt = get_test_data("caida")

    cls = RandomForestClassifier(n_estimators=100).fit(x, y)
    cls1 = RandomForestClassifier(n_estimators=10).fit(x, y)
    cls2 = RandomForestClassifier(n_estimators=10).fit(x, y)
    cls3 = RandomForestClassifier(n_estimators=10).fit(x, y)

    # extract_n_trees(cls.estimators_, x, y, 'acc', lower=False, num_features=18)

    fed_acc = FedForest(xt, yt, 50).fit_acc(cls.estimators_)
    fed_bic = FedForest(xt, yt, 50).fit_bic(cls.estimators_, 18)
    fed_bic_high = FedForest(xt, yt, 50).fit_bic(cls.estimators_, 18, False)

    print("Fed: ", np.sum(fed_acc.predict(xt) == yt))
    cls1.estimators_ = extract_n_trees(cls.estimators_, 50, xt, yt, 'acc')
    print("Ext: ", np.sum(cls1.predict(xt) == yt))

    print("Fed: ", np.sum(fed_bic.predict(xt) == yt))
    cls2.estimators_ = extract_n_trees(cls.estimators_, 50, xt, yt, 'bic', lower=True, num_features=18)
    print("Ext: ", np.sum(cls2.predict(xt) == yt))

    print("Fed: ", np.sum(fed_bic_high.predict(xt) == yt))
    cls3.estimators_ = extract_n_trees(cls.estimators_, 50, xt, yt, 'bic', lower=False, num_features=18)
    print("Ext: ", np.sum(cls3.predict(xt) == yt))

    # print(np.sum(cls.predict(xt) == yt))
