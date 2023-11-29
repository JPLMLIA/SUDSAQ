import logging
import multiprocessing as mp
import numpy as np
import warnings

mp.set_start_method('fork')
warnings.simplefilter(action='ignore', category=UserWarning)

from functools    import partial
from sklearn.tree import _tree

Logger = logging.getLogger(__file__)


def get_paths(tree, node=0):
    """
    Traverses a DecisionTree and returns a dictionary where the keys are the leaf nodes,
    and the values are lists representing the index path from the root node to each leaf.

    Parameters
    ----------
    tree : DecisionTree object
        The DecisionTree to be traversed.
    node : int, optional (default=0)
        The starting node for traversal. Defaults to the root node (0).

    Returns
    -------
    dict
        A dictionary where keys are leaf nodes, and values are lists representing
        the index path from the root node to each leaf.
    """
    def traverse(node):
        """
        Recursively traverses the DecisionTree and returns a list of paths from the
        given node to all leaf nodes.

        Parameters
        ----------
        node : int
            The current node being traversed.

        Returns
        -------
        list
            A list of paths from the current node to all leaf nodes.
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

    # Find all possible paths and convert to a dict of the {leaf node: path list}
    paths = traverse(node)

    return {path[-1]: path for path in paths}


def predict_tree(estimator, X=None):
    """
    Predicts outcomes for a single tree model and calculates biases and feature
    contributions.

    Parameters
    ----------
    estimator : DecisionTreeRegressor or DecisionTreeClassifier
        The tree model to calculate predictions, biases, and contributions from.
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    tuple
        A tuple containing three arrays:
        - predicts : array-like
            Direct predictions for each sample.
        - biases : array-like
            Biases for each prediction.
        - contributions : array-like
            Feature contributions for each prediction.

    Notes
    -----
    The contributions are calculated based on the minimal feature contribution paths
    in the tree.
    """
    # To save memory space, global variables are shared memory in multiprocessing
    if isinstance(estimator, int):
        estimator = MODEL.estimators_[estimator]
    if X is None:
        X = DATA

    tree    = estimator.tree_
    paths   = get_paths(tree, 0)
    leaves  = estimator.apply(X)
    uniques = np.unique(leaves)
    values  = tree.value.ravel()

    # Maps the nodes of the tree to the feature index
    features = list(tree.feature)

    predicts = values[leaves]
    biases   = np.full_like(predicts, values[0])
    contribs = {}

    # Calculate the minimal feature contribution paths (uniques)
    cache = {}
    for i, leaf in enumerate(uniques):
        path = paths[leaf]
        cont = contribs.setdefault(i, {})
        for j in range(len(path) - 1):
            root  = path[j]
            child = path[j + 1]
            feature = features[root]
            cont.setdefault(feature, 0)
            cont[feature] += values[child] - values[root]

    # Map the leaf value to its index in the uniques list
    index = {leaf: i for i, leaf in enumerate(uniques)}

    # Mask represents for each sample which index from the uniques list to copy as its contributions
    mask = [index[leaf] for leaf in leaves]

    # Calculate the full contribution object
    full = [contribs[index] for index in mask]

    return predicts, biases, full


def updateConts(contributions, update):
    """
    Updates a contributions list with another

    Parameters
    ----------
    contributions: list
        Contribution list containing dicts which represent the
        {feature number: contribution value} for each sample
    update: list
        Same as `contributions`, this will update that
    """
    # Sample is the minimum set of feature importance for that sample index
    for i, sample in enumerate(update):
        # Update the overall contribution for this sample
        overall = contributions.setdefault(i, {})
        for feature, value in sample.items():
            overall.setdefault(feature, 0)
            overall[feature] += value


def predict_forest(model, X, n_jobs=None):
    """
    Predicts outcomes for an entire forest model and calculates biases and feature contributions.

    Parameters
    ----------
    model : RandomForestRegressor
        The random forest model to make predictions with.
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        The input data.
    n_jobs : int, optional (default=None)
        The number of parallel jobs to run. Follows the style of sklearn's n_jobs parameter:
              -1 : All available cores
            None : 1 core
             int : Number of cores to use.

    Returns
    -------
    tuple
        A tuple containing three arrays:
        - predicts : array-like
            Direct predictions for each sample.
        - biases : array-like
            Biases for each prediction.
        - contributions : array-like
            Feature contributions for each prediction.

    Notes
    -----
    The contributions are calculated by aggregating individual tree contributions.
    """
    # Pool(processes=None) is all cores
    n_jobs = {-1: None, None: 1}.get(n_jobs, n_jobs)

    predicts = np.zeros(X.shape[0])
    biases   = np.zeros_like(predicts)
    contribs = {}

    # Insert the entire model into shared memory so its not duplicated
    global MODEL
    MODEL = model

    # Insert X into global space so that it is shared by processes instead of copied
    global DATA
    DATA = X

    Logger.debug('Calculating contributions')

    total = len(model.estimators_)
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.imap_unordered(predict_tree, list(range(total)))
        for i, (p, b, c) in enumerate(results):
            predicts += p
            biases   += b
            updateConts(contribs, c)

            Logger.debug(f'{i}/{total} ({i/total:.2f}%) completed')

    Logger.debug('Calculating average')
    # Divide by total for average
    predicts /= total
    biases   /= total
    for sample in contribs.values():
        for feature in sample:
            sample[feature] /= total

    Logger.debug('Creating contributions array object')
    # Convert from the list of dicts form to a 2D array
    contributions = np.zeros_like(X)
    for sample, features in contribs.items():
        values   = list(features.values())
        features = list(features)
        contributions[sample, features] = values

    # Reshape to the original TreeInterpreter shape
    predicts = predicts.reshape(-1, 1)

    Logger.debug('Calculations done')
    return predicts, biases, contributions


def predict(model, X, n_jobs=None):
    """
    Returns a triple (prediction, bias, feature_contributions), such that
    prediction ≈ bias + feature_contributions

    Parameters
    ----------
    model: ...
        Scikit-learn model on which the prediction should be decomposed.

    X: array-like, shape=(n_samples, n_features)
        Test samples.

    Returns
    -------
    decomposed prediction: tuple of
        * prediction, shape = (n_samples) for regression
        * bias, shape = (n_samples) for regression
        * contributions, array of
            shape = (n_samples, n_features) for regression
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError('Multilabel classification trees not supported')

    return predict_forest(model, X, n_jobs=n_jobs)
