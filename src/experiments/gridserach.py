import pickle

from src.utils.pathtools import project
from src.utils.logs import logger
from src.experiments.datasets import diatom_ds, kitchen_ds, pressure_ds
from src.experiments.laplacian_score import compute_weight_matrix, laplacian_score, get_features_to_keep_laplacian, dtw_pairwise_distances
from src.experiments.classifiers import get_svc_accuracy

DATASETS = [diatom_ds, kitchen_ds, pressure_ds]

# DTW pairwise distance matrix
DISK_FILE = 'dtw_distance_matrix_{dataset}.pkl'
distance_matrixes = dict()

for ds in DATASETS:
    logger.info(f'Getting the DTW distance matrix with dataset {ds.name}')
    # Looking on the disk
    file_path = project.output / DISK_FILE.format(dataset = ds.name)
    if file_path.exists():
        with file_path.open('rb') as f:
            distance_matrixes[ds.name] = pickle.load(f)
    else:
        distance_matrixes[ds.name] = dtw_pairwise_distances(ds)
        with file_path.open('wb') as f:
            pickle.dump(distance_matrixes[ds.name], f)
