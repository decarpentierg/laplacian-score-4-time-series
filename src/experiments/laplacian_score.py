import typing as t

import numpy as np
import pandas as pd

from sklearn.neighbors import kneighbors_graph

from src.experiments.datasets import Dataset
from src.experiments.pairwise_distances import dtw_pairwise_distances
from src.utils.logs import logger


def compute_weight_matrix(
        X,
        n_neighbors: int = 5,
        sigma: float = 1.0,
        precomputed_distances: np.ndarray = None
    ) -> np.ndarray:
    """Compute weight matrix of an array of time series.

    Parameters
    ----------
    X: ndarray of shape (m, T) or Dataset object or Pandas DataFrame object
        Dataset of time series
    
    n_neighbors: int
        number of nearest neighbors to consider
    
    sigma: float
        corresponds to parameter t in the article

    precomputed_distances: np.ndarray
        an array of shape (m, m) with the precomputed distances between the series

    Returns
    ----------
    S: np.ndarray
        an array of shape (m, m) corresponding to the weight matrix.
    
    References
    ----------
    [1] He et al., "Laplacian Score for Feature Selection", 2005
    """
    # Convert to np.ndarray
    if isinstance(X, Dataset):
        X = np.array(X.data.iloc[:, :-1])
    if isinstance(X, pd.DataFrame):
        X = np.array(X.iloc[:, :-1])
    else:
        assert isinstance(X, np.ndarray)

    logger.info(f'Comupting weight matrix for a ds of shape {X.shape}: n_neighbors={n_neighbors}, sigma={sigma}')
    
    # Compute pairwise distances
    if precomputed_distances is None:
        pw_dist = dtw_pairwise_distances(X)
    else:
        pw_dist = precomputed_distances

    # Compute k-neighbors graph
    adjacency_matrix = kneighbors_graph(X, n_neighbors)

    # Compute weight matrix
    S = adjacency_matrix * np.exp(-pw_dist**2 / (2 * sigma**2))

    return S


def laplacian_score(f, S) -> t.List[float]:
    """Compute Laplacian score.

    Parameters
    ----------
    f: ndarray, shape (n_features, m) or Dataset object
        Matrix with the n_features features of the m data points.

    S: ndarray, shape (m, m)
        Weight matrix

    Returns
    ----------
    scores: t.List[float]
        The list of laplacian score of the features

    References
    ----------
    [1] He et al., "Laplacian Score for Feature Selection", 2005
    """
    # Convert to np.ndarray
    if isinstance(f, Dataset):
        f = f.features
    else:
        assert isinstance(f, np.ndarray)

    logger.info(f'Computing the Laplacian Score of {f.shape[0]} features belonging to {f.shape[1]} time series')

    d = np.sum(S, axis=1)  # diagonal of matrix D in the article, shape=(m,)
    D = np.diag(d)  # shape=(m, m)
    L = D - S  # shape=(m, m)
    mean = np.sum(f * d, axis=1) / np.sum(d) # shape=(m,)
    f_tilde = f - mean[:, np.newaxis]  # shape=(n_features, m)
    num   = np.einsum('ri,rj,ij->r', f_tilde, f_tilde, L)  # numerator, shape=(n_features,)
    denom = np.einsum('ri,rj,ij->r', f_tilde, f_tilde, D)  # denominator, shape=(n_features,)
    denom[denom==0.] = 1e-10  # to avoid division by zero
    return num / denom  # shape=(n_features,)


class LaplacianSelection(object):

    def __init__(
            self,
            ds_name: str,
            n_features: int,
            use_dtw: bool = True,
            **weight_matrix_kwargs
        ) -> None:
        """Initiates a Feature Selection object, usable in a sklearn.Pipeline.

        Parameters
        ----------
        ds_name: str
            The name of the dataset for which this feature selector is built

        n_features: int
            The number of features to keep

        use_dtw: int
            If true, the DTW distance will be used. Else, it will be the euclidian distance.

        weight_matrix_kwargs: Any
            Kwargs to be passed to compute_weight_matrix, such as `sigma` or `n_neighbors`.
        
        """
        self._dataset = Dataset(ds_name)
        self._n_features = n_features
        
        # Precomputed distances between the series 
        if use_dtw:
            self._precomputed_distances = self._dataset.dtw_distance_matrix
        else:
            self._precomputed_distances = self._dataset.ned_distance_matrix

        # Weight matrix
        self._weight_matrix = compute_weight_matrix(
            self._dataset,
            precomputed_distances=self._precomputed_distances,
            **weight_matrix_kwargs,
        )

        # Laplacian scores
        self._laplacian_scores = laplacian_score(
            self._dataset,
            self._weight_matrix,
        )

    def fit(self, X:np.ndarray, y:np.ndarray=None):
        return self
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        """Returns the array, keeping only the relevant columns."""
        logger.info(f'Using Laplacian score to select features from dataset {self._dataset.name}')
        columns_to_keep = sorted(
            range(len(self._laplacian_scores)),
            key=lambda i: self._laplacian_scores[i],
            reverse=True
        )[:self._n_features]

        return X[columns_to_keep]
