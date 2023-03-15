from datetime import datetime
import typing as t

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

from src.utils.logs import logger
from src.utils.pathtools import project
from src.experiments.datasets import kitchen_ds, diatom_ds, pressure_ds
from src.experiments.laplacian_score import LaplacianSelection
from src.experiments.classifiers import get_knn_accuracy, get_svc_accuracy


def get_features_labels_and_selectors() -> t.Dict[str, t.Tuple[np.ndarray, np.ndarray, t.Any]]:
    result = dict()
    for ds in [kitchen_ds, diatom_ds, pressure_ds]:

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
                                sigma = kitchen_ds.dtw_distance_matrix.mean() * sigma_factor,
                                n_neighbors = num_neighbors,
                            ),
                        )

            # ANOVA F-value feature selection
            result[f'{ds.name}_fclassif_nfeat={n_features}'] = (
                ds.features.T,
                ds.labels,
                SelectKBest(f_classif, k=n_features) #.fit_transform(ds.features.T, ds.labels),
            )


        # Multiple parameters for Variance feature selection
        for threshold in [0.001, 0.01, 0.1, 1]:
            result[f'{ds.name}_variance_threshold={threshold}'] = (
                ds.features.T,
                ds.labels,
                VarianceThreshold(threshold=threshold)
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
