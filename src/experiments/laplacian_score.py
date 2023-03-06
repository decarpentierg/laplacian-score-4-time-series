from dtw import dtw
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from src.experiments.datasets import Dataset
from src.utils.logs import logger

def dtw_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Computes the DTW pairwise distance between time series.
    
    Parameters
    ----------
    X: ndarray of shape (m, T) or Dataset object or Pandas DataFrame object
        Dataset of time series
    """

    # Convert to np.ndarray
    if isinstance(X, Dataset):
        X = np.array(X.data.iloc[:, :-1])
    if isinstance(X, pd.DataFrame):
        X = np.array(X.iloc[:, :-1])
    else:
        assert isinstance(X, np.ndarray)

    n_signals = X.shape[0]
    result = np.zeros((n_signals, n_signals))
    logger.info(f'Computing pairwise DTW distance between {n_signals} time series')    
    
    for idx_0 in tqdm(range(n_signals)):
        for idx_1 in range(idx_0+1, n_signals):
            dst = dtw(X[idx_0, :], X[idx_1, :]).distance
            result[idx_0, idx_1] = dst
            result[idx_1, idx_0] = dst

    return result


def compute_weight_matrix(X, n_neighbors: int = 5, sigma: float = 1.0, use_dtw: bool = True):
    """Compute weight matrix of an array of time series.

    Parameters
    ----------
    X: ndarray of shape (m, T) or Dataset object or Pandas DataFrame object
        Dataset of time series
    
    n_neighbors: int
        number of nearest neighbors to consider
    
    sigma: float
        corresponds to parameter t in the article

    use_dtw: bool
        whether or not to use DTW insead of euclidian distance
    
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

    logger.info(f'Comupting weight matrix for a ds of shape {X.shape}: n_neighbors={n_neighbors}, sigma={sigma}, use_dtw={use_dtw}')
    
    # Compute pairwise distances
    if use_dtw:
        pw_dist = dtw_pairwise_distances(X)
    else:
        pw_dist = pairwise_distances(X)

    # Compute k-neighbors graph
    adjacency_matrix = kneighbors_graph(X, n_neighbors)

    # Compute weight matrix
    S = adjacency_matrix * np.exp(-pw_dist**2 / (2 * sigma**2))

    return S


def laplacian_score(f, S):
    """Compute Laplacian score.

    Parameters
    ----------
    f: ndarray, shape (n_features, m) or Dataset object
        Matrix with the n_features features of the m data points.

    S: ndarray, shape (m, m)
        Weight matrix

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
    return num / denom  # shape=(n_features,)


if __name__ == '__main__':
    from src.experiments.datasets import heartbeat_ds, diatom_ds, pressure_ds
    dataset = diatom_ds
    n_neighbors = 5
    sigma = 0.1
    S = compute_weight_matrix(dataset, n_neighbors=n_neighbors, sigma=sigma)
    scores = laplacian_score(dataset, S)
    print(sorted(scores))