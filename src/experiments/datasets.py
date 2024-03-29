"""
Datasets:
-   SmallKitchenAppliances [1]
-   DiatomSizeReduction [2]
-   PigAirwayPressure [3]

==============================

References
----------
-   [1] Jason Lines, A. Bagnall, https://timeseriesclassification.com/description.php?Dataset=SmallKitchenAppliances
-   [2] https://timeseriesclassification.com/description.php?Dataset=DiatomSizeReduction
-   [3] M. Guillame-Bert, https://timeseriesclassification.com/description.php?Dataset=PigAirwayPressure
"""

import sys
import zipfile
import pickle

import pandas as pd
import numpy as np
from scipy.io import arff
import requests
from tqdm import tqdm
import tsfel

from src.utils.pathtools import project
from src.utils.logs import logger

from src.experiments.pairwise_distances import dtw_pairwise_distances, ned_pairwise_distances
from src.utils.constants import *


DATASETS_DOWNLOAD_LINK = {
    name:f'https://timeseriesclassification.com/Downloads/{name}.zip'
    for name in [KITCHEN, DIATOM, PRESSURE, EARTHQUAKES, WAFER, WORMS]
}
DTW_DM_PATH = 'dtw_distance_matrix_{dataset}.pkl'
NED_DM_PATH = 'ned_distance_matrix_{dataset}.pkl'
FEATURES_PATH = 'features_{dataset}.pkl'
LABELS_COLUMN = 'target'
DEFAULT_MAX_SAMPLE = 2000


class Dataset():

    def __init__(self, name:str, max_sample=DEFAULT_MAX_SAMPLE) -> None:
        logger.info(f'Initiating dataset {name}')
        logger.info(f'More info at https://timeseriesclassification.com/description.php?Dataset={name}')
        self.name = name
        self._data = None
        self._labels = None
        self._features = None
        self._dtw_distance_matrix = None
        self._ned_distance_matrix = None
        self._dir_path = project.data / self.name
        self._zip_path = project.data / f'{self.name}.zip'
        self._max_sample = max_sample

    @property
    def data(self) -> pd.DataFrame:
        """Returns a pd.DataFrame of shape (m, T) 
        where m is the number of time series and T the number of timestamp in the series."""
        if self._data is None:
            self._get_data()
        return self._data
    
    @property
    def labels(self) -> np.array:
        """Returns a np.ndarray of shape (m) where m is the number of time series.
        The dtype of this array is `int`."""
        if self._labels is None:
            self._labels = np.asarray(self.data[LABELS_COLUMN].values, dtype=int)
        return self._labels
    
    @property
    def features(self) -> np.ndarray:
        """Returns a np.ndarray of shape (n_features, m)
        where m is the number of time series"""
        if self._features is None:
            self._get_features()
        return self._features

    @property
    def dtw_distance_matrix(self) -> np.ndarray:
        """Returns an np.ndarray of shape (m, m), where m is the number of time series."""
        if self._dtw_distance_matrix is None:
            self._get_dtw_distance_matrix()
        return self._dtw_distance_matrix

    @property
    def ned_distance_matrix(self) -> np.ndarray:
        """Returns an np.ndarray of shape (m, m), where m is the number of time series."""
        if self._ned_distance_matrix is None:
            self._get_ned_distance_matrix()
        return self._ned_distance_matrix
    
    def __len__(self):
        return len(self.data.index)
    
    # -------------------- GET DATA ---------------------

    def _get_data(self):
        """Gets data and load it in self._data"""
        self._check_downloaded()
        self._load_data()
        self._clip_data()

    def _check_downloaded(self):
        """Checks that the data is correctly downloaded"""

        if not self._dir_path.exists() or len(list(self._dir_path.iterdir())) <= 2:
            logger.debug(f'{self.name} dataset not found')

            if not self._zip_path.exists():
                logger.info(f'{self.name}  dataset zip not found, downloading it...')
                response = requests.get(DATASETS_DOWNLOAD_LINK[self.name], stream=True)
                with self._zip_path.open('wb') as f:
                    dl = 0
                    total_length = response.headers.get('content-length')
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\rProgression: [%s%s]" % ('=' * done, ' ' * (50-done)) )    
                        sys.stdout.flush()

                sys.stdout.write('\n')

            logger.debug(f'Extracting {self.name}...')
            try:
                with zipfile.ZipFile(self._zip_path) as zf:
                    zf.extractall(project.mkdir_if_not_exists(self._dir_path))
            except zipfile.BadZipFile:
                logger.debug(f'Found corrupted .zip file, deleting it and trying again...')
                self._zip_path.unlink()
                self._check_downloaded()

        else:
            logger.debug(f'{self.name} found at {project.as_relative(self._dir_path)}')

    def _load_data(self):
        """Loads data in `self.data`"""
        logger.debug(f'Loading dataset from disk: {self.name}')
        data_train_arff = arff.loadarff(self._dir_path / f'{self.name}_TRAIN.arff')
        data_train_df = pd.DataFrame(data_train_arff[0])
        data_test_arff = arff.loadarff(self._dir_path / f'{self.name}_TEST.arff')
        data_test_df = pd.DataFrame(data_test_arff[0])
        
        self._data = pd.concat(
            [data_train_df, data_test_df],
            ignore_index=True,
        )

    def _clip_data(self):
        """Deletes the final samples if there are too many of them."""
        if self._data.shape[0] > self._max_sample:
            self._data = self._data[:self._max_sample]

    # -------------------- GET FEATURES AND DISTANCE MATRICES ---------------------

    def _get_features(self):
        """Extracts the features with tsfel"""
        logger.info(f'Extracting the features of {len(self)} time series')
        file_path = project.saved_dataset_attributes / FEATURES_PATH.format(dataset = self.name)
        if file_path.exists():  # check if file exists
            with file_path.open('rb') as f:
                self._features = pickle.load(f)
        else:  
            # if not, compute features
            result = list()
            for index in tqdm(range(len(self))):
                serie = self.data.iloc[index, :-1]
                cfg = tsfel.get_features_by_domain()
                result.append(np.squeeze(
                    tsfel.time_series_features_extractor(cfg, serie, verbose=0).values
                ))
            # assign to attribute features
            self._features = np.array(result).transpose()
            # check features shape
            assert self._features.shape[1] == len(self), f'Incorrect features shape: {self._features.shape[1]} != {len(self)}'
            # save features
            with file_path.open('wb') as f:
                pickle.dump(self._features, f)
    
    def _get_dtw_distance_matrix(self):
        """Compute the DTW distance matrix"""
        logger.debug(f'Getting the DTW distance matrix with dataset {self.name}')
        # Looking on the disk
        file_path = project.saved_dataset_attributes / DTW_DM_PATH.format(dataset = self.name)
        if file_path.exists():
            with file_path.open('rb') as f:
                self._dtw_distance_matrix = pickle.load(f)
        else:
            self._dtw_distance_matrix = dtw_pairwise_distances(np.array(self.data.iloc[:, :-1]))
            with file_path.open('wb') as f:
                pickle.dump(self._dtw_distance_matrix, f)
    
    def _get_ned_distance_matrix(self):
        """Compute the NED distance matrix"""
        logger.debug(f'Getting the NED distance matrix with dataset {self.name}')
        # Looking on the disk
        file_path = project.saved_dataset_attributes / NED_DM_PATH.format(dataset = self.name)
        if file_path.exists():
            with file_path.open('rb') as f:
                self._ned_distance_matrix = pickle.load(f)
        else:
            self._ned_distance_matrix = ned_pairwise_distances(np.array(self.data.iloc[:, :-1]))
            with file_path.open('wb') as f:
                pickle.dump(self._ned_distance_matrix, f)

earthquakes_ds = Dataset(EARTHQUAKES)
wafer_ds = Dataset(WAFER)
worms_ds = Dataset(WORMS)

if __name__ == '__main__':
    # Builds the datasets cache
    _ = earthquakes_ds.features
    _ = earthquakes_ds.dtw_distance_matrix
    _ = earthquakes_ds.ned_distance_matrix
    _ = wafer_ds.features
    _ = wafer_ds.dtw_distance_matrix
    _ = wafer_ds.ned_distance_matrix
    _ = worms_ds.features
    _ = worms_ds.dtw_distance_matrix
    _ = worms_ds.ned_distance_matrix

    # Same for previous DS
    _ = Dataset(KITCHEN).features
    _ = Dataset(KITCHEN).dtw_distance_matrix
    _ = Dataset(KITCHEN).ned_distance_matrix
    _ = Dataset(PRESSURE).features
    _ = Dataset(PRESSURE).dtw_distance_matrix
    _ = Dataset(PRESSURE).ned_distance_matrix
    _ = Dataset(DIATOM).features
    _ = Dataset(DIATOM).dtw_distance_matrix
    _ = Dataset(DIATOM).ned_distance_matrix