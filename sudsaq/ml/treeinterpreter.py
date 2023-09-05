# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
import numpy as np
import sklearn
import warnings

from distutils.version import LooseVersion
from sklearn.ensemble  import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    _tree
)

Logger = logging.getLogger('treeinterpreter.py')

if LooseVersion(sklearn.__version__) < LooseVersion('0.17'):
    raise Exception('treeinterpreter requires scikit-learn 0.17 or later')

warnings.simplefilter(action='ignore', category=UserWarning)


def get_paths(tree, node):
    """
    """
    def traverse(node):
        """
        """
        if node == _tree.TREE_LEAF:
            raise ValueError(f'Invalid node id: {node}')

        left  = tree.children_left[node]
        right = tree.children_right[node]

        if left == _tree.TREE_LEAF:
            return [(node,),]

        paths = traverse(left) + traverse(right)

        # Prepend instead of append to reverse the path in place
        for i, path in enumerate(paths):
            paths[i] = (node, *path)

        return paths

    # Initialize the cache as an attribute of this function for persistency
    if not hasattr(get_paths, 'cache'):
        get_paths.cache = {}

    key = f'{tree}{node}'
    if key not in get_paths.cache:
        paths = traverse(node)
        get_paths.cache[key] = {path[-1]: path for path in paths}

    return get_paths.cache[key]

def predict_tree(estimator, X):
    """
    """
    tree     = estimator.tree_
    paths    = get_paths(tree, 0)
    leaves   = estimator.apply(X)
    uniques  = np.unique(leaves)

    values   = tree.value.squeeze(axis=1)
    predicts = values[leaves]
    values   = list(values) # Slightly faster lookups

    # Maps the nodes of the tree to the feature index
    features = list(tree.feature)

    # Index is the feature contribution for that leaf column
    contribs = np.zeros((len(uniques), X.shape[1]))
    contribs[:] = 0

    biases = np.full_like(predicts, values[0])
    for i, leaf in enumerate(uniques):
        path = paths[leaf]
        for j in range(len(path) - 1):
            # Update the contributions for this leaf/path per feature
            contribs[i, features[j]] += values[path[j+1]] - values[path[j]]

    index = {leaf: i for i, leaf in enumerate(uniques)}
    contributions = np.array([contribs[index[leaf]] for leaf in leaves])

    return predicts, biases, contributions

def predict_forest(model, X, n_jobs=None):
    """
    Parameters
    ----------
    model
    X
    n_jobs: int, defaults=None
        Similar to sklearn n_jobs:
              -1 = all cores
            None = 1 core
             int = N many cores
    """
    # processes=None == all cores
    n_jobs = {-1: None, None: 1}.get(n_jobs, n_jobs)

    predicts      = np.zeros((X.shape[0], 1))
    biases        = np.zeros_like(predicts)
    contributions = np.zeros_like(X)

    total = len(model.estimators_)
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.imap_unordered(predict_tree, model.estimators_)
        for i, (p, b, c) in enumerate(results):
            predicts      += p
            biases        += b
            contributions += c

            Logger.debug(f'{i}/{total} ({i/total:.2f}%) completed')

    predicts      /= total
    biases        /= total
    contributions /= total

    return predicts, biases, contributions

def predict(model, X, n_jobs=None):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.

    Parameters
    ----------
    model: DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor,
        ExtraTreeClassifier, RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier

        Scikit-learn model on which the prediction should be decomposed.

    X: array-like, shape=(n_samples, n_features)

        Test samples.

    Returns
    -------
    decomposed prediction : triple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, array of
        shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification, denoting
        contribution from each feature.
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError('Multilabel classification trees not supported')

    if isinstance(model, (
        DecisionTreeClassifier,
        DecisionTreeRegressor
    )):
        return predict_tree(model, X)
    elif isinstance(model, (
        RandomForestClassifier,
        ExtraTreesClassifier,
        RandomForestRegressor,
        ExtraTreesRegressor,
    )):
        return predict_forest(model, X, n_jobs=n_jobs)
    else:
        raise ValueError('Wrong model type. Base learner needs to be a DecisionTreeClassifier or DecisionTreeRegressor.')
