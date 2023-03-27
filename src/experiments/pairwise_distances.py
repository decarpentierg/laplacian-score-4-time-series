import numpy as np
from tqdm import tqdm

from dtw import dtw
from sklearn.metrics import pairwise_distances

from src.utils.logs import logger


def dtw_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Computes the DTW pairwise distance between time series.
    
    Parameters
    ----------
    X: ndarray of shape (m, T)
        Dataset of time series

    Returns
    ----------
    distances
        an array of shape (m, m) containing the pairwise DTW distances
    """

    n_signals = X.shape[0]
    signals_len = X.shape[1]
    result = np.zeros((n_signals, n_signals))
    logger.info(f'Computing pairwise DTW distance between {n_signals} time series')    
    
    # sakoechiba for too long series
    if signals_len > 500:
        logger.debug(f'Too long series, switching to sakoechiba windows for the DTW')
        dtw_kwargs = {
            'window_type':'sakoechiba',
            'window_args':{'window_size':100},
        }
    else:
        dtw_kwargs = dict()

    for idx_0 in tqdm(range(n_signals)):
        for idx_1 in range(idx_0+1, n_signals):
            dst = dtw(X[idx_0, :], X[idx_1, :], distance_only=True, **dtw_kwargs).distance
            result[idx_0, idx_1] = dst
            result[idx_1, idx_0] = dst

    return result


def ned_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Computes the NED pairwise distances between time series.
    NED = Normalized Euclidian Distance
    
    Parameters
    ----------
    X: ndarray of shape (m, T)
        Dataset of time series

    Returns
    -------
    distances
        an array of shape (m, m) containing the pairwise NED distances
    """

    m = X.shape[0]
    logger.info(f'Computing pairwise NED distance between {m} time series')
    Xn = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return pairwise_distances(Xn)
