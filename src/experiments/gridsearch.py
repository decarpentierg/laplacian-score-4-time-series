from datetime import datetime
import typing as t

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

from src.utils.logs import logger
from src.utils.pathtools import project
from src.experiments.datasets import Dataset, earthquakes_ds, wafer_ds, worms_ds
from src.experiments.laplacian_score import LaplacianSelection
from src.experiments.classifiers import get_knn_accuracy, get_svc_accuracy


class VarianceSelection(object):

    """Similar to `sklearn.feature_selection.VarianceThreshold`, but using
    a number of features instead of a variance threshold."""

    def __init__(
        self,
        dataset: Dataset,
        n_features: int,
    ) -> None:
        
        self.dataset = dataset
        self.n_features = n_features

    def fit(self, X:np.ndarray, y:np.ndarray=None):
        self.variance = np.nanvar(self.dataset.features, axis = 1)
        self.threshold = np.partition(self.variance, -self.n_features)[-self.n_features]
        return self
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        return X[:, self.variance >= self.threshold]
    

def get_variance_threshold(
    dataset: Dataset,
    n_features: int,
) -> float:
    """Compute a variance threshold so that there is only the desired number of features
    whose variance are greater or equal.
    
    :param dataset: The dataset
    :param n_features: The desired number of features
    :returns: The variance threshold"""

    variance = np.var(dataset.features, axis = 1)
    return np.partition(variance, -n_features)[-n_features]


def get_features_labels_and_selectors() -> t.Dict[str, t.Tuple[np.ndarray, np.ndarray, t.Any]]:
    result = dict()
    for ds in [earthquakes_ds, wafer_ds, worms_ds]:

        for n_features in [10, 50, 100, 150]:

            # Multiple parameters for Laplacian feature selection
            for use_dtw in [True, False]:
                for sigma_factor in [0.001, 0.01,0.1, 1]:
                    for num_neighbors in [5,10,20,30]:
                        result[f'{ds.name}_laplacian_nfeat={n_features}_dtw={use_dtw}_sigma={sigma_factor}_neigh={num_neighbors}'] = (
                            ds.features.T,
                            ds.labels,
                            LaplacianSelection(
                                ds,
                                n_features=n_features,
                                use_dtw=use_dtw,
                                sigma = ds.dtw_distance_matrix.mean() * sigma_factor,
                                n_neighbors = num_neighbors,
                            ),
                        )

            # ANOVA F-value feature selection
            result[f'{ds.name}_fclassif_nfeat={n_features}'] = (
                ds.features.T,
                ds.labels,
                SelectKBest(f_classif, k=n_features)
            )


        # Multiple parameters for Variance feature selection
        for threshold in [0.001, 0.01, 0.1, 1]:
            result[f'{ds.name}_variance_threshold={threshold}'] = (
                ds.features.T,
                ds.labels,
                VarianceThreshold(threshold=get_variance_threshold(ds, n_features))
            )

        # No feature selection 
        result[f'{ds.name}_no_selection'] = (
            ds.features.T,
            ds.labels,
            None,
        )

    return result


def main():
    configs = get_features_labels_and_selectors()
    n_configs = len(configs)
    result = np.zeros((n_configs, 1+2*3), dtype=object)

    logger.info(f'Evaluating {n_configs} configurations')
    for index, (config_name, (features, labels, selector)) in enumerate(tqdm(configs.items())):
        result[index, 0] = config_name
        result[index, 1], result[index, 2] = get_svc_accuracy(features, labels, selector)
        result[index, 3], result[index, 4] = get_knn_accuracy(features, labels, selector, n_neighbors = 5)
        result[index, 5], result[index, 6] = get_knn_accuracy(features, labels, selector, n_neighbors = 10)

    result = pd.DataFrame(
        result,
        columns=['Config','SVC precision', 'SVC recall', 'KNN-5 precision', 'KNN-5 recall', 'KNN-10 precision', 'KNN-10 recall']
    )
    
    result.to_csv(
        project.output / datetime.now().strftime("gridsearch_%Y_%m_%d__%H_%M_%S.csv")
    )

    return result

if __name__ == '__main__':
    main()
