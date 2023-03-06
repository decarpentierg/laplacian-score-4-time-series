import typing as t

import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_svc_accuracy(features: np.ndarray, labels:np.ndarray, **svc_kwargs) -> t.Tuple[float, float]:
    """Computes the accuracy of the SVC classifier over the features.
    Half of the dataset is used for the training, and half for the testing.
    
    :param features: The features, an array of shape (m, n_features)
    :param labels: The labels, an array of shape (m, 1)
    :param svc_kwargs: Kwargs for sklearn.svm.SVC
    :returns: The tupple (precision, recall)
    """
    m = features.shape[0]
    clf = make_pipeline(StandardScaler(), SVC(**svc_kwargs))
    clf.fit(features[:m//2], labels[:m//2])
    preds = clf.predict(features[m//2:])
    acc = accuracy_score(labels[m//2:], preds)
    rec = recall_score(labels[m//2:], preds)

    return acc, rec
