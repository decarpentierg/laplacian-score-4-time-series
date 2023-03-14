import typing as t

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC


def get_svc_accuracy(
        features: np.ndarray,
        labels:np.ndarray,
        feature_selector = None,
        **clf_kwargs
) -> t.Tuple[float, float]:
    """Computes the accuracy of the SVC classifier over the features.
    Half of the dataset is used for the training, and half for the testing.
    
    :param features: The features, an array of shape (m, n_features)
    :param labels: The labels, an array of shape (m, 1)
    :param feature_selector: An object that can perform feature selection within
    a sklearn pipeline, such as sklearn.feature_selection.SelectFromModel.
    This object should have at least two methods: `self.fit(X, y)` and `self.transform(X) -> np.ndarray`.
    :param clf_kwargs: Kwargs for sklearn.svm.SVC
    :returns: The tupple (precision, recall)
    """
    m = features.shape[0]
    
    clf = make_pipeline(
        StandardScaler(),
        feature_selector,
        SVC(**clf_kwargs),
    )
    clf.fit(features[:m//2], labels[:m//2])
    preds = clf.predict(features[m//2:])
    acc = accuracy_score(labels[m//2:], preds)
    rec = recall_score(labels[m//2:], preds, average='weighted', zero_division=1)

    return acc, rec


def get_knn_accuracy(
        features: np.ndarray,
        labels:np.ndarray,
        feature_selector = None,
        **clf_kwargs
) -> t.Tuple[float, float]:
    """Computes the accuracy of the SVC classifier over the features.
    Half of the dataset is used for the training, and half for the testing.
    
    :param features: The features, an array of shape (m, n_features)
    :param labels: The labels, an array of shape (m, 1)
    :param feature_selector: An object that can perform feature selection within
    a sklearn pipeline, such as sklearn.feature_selection.SelectFromModel.
    This object should have at least two methods: `self.fit(X, y)` and `self.transform(X) -> np.ndarray`.
    :param clf_kwargs: Kwargs for sklearn.neighbors.KNeighborsClassifier
    :returns: The tupple (precision, recall)
    """
    m = features.shape[0]

    clf = make_pipeline(
        StandardScaler(),
        feature_selector,
        KNeighborsClassifier(**clf_kwargs),
    )
    clf.fit(features[:m//2], labels[:m//2])
    preds = clf.predict(features[m//2:])
    acc = accuracy_score(labels[m//2:], preds)
    rec = recall_score(labels[m//2:], preds, average='weighted', zero_division=1)

    return acc, rec
