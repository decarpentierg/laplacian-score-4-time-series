"""
Datasets:
-   AbnormalHeartbeat [1]
-   DiatomSizeReduction [2]
-   PigAirwayPressure [3]

==============================

References
----------
-   [1] UCR, Michael Flynn, https://timeseriesclassification.com/description.php?Dataset=AbnormalHeartbeat
-   [2] https://timeseriesclassification.com/description.php?Dataset=DiatomSizeReduction
-   [3] M. Guillame-Bert, https://timeseriesclassification.com/description.php?Dataset=PigAirwayPressure
"""

import sys
import zipfile

import pandas as pd
import numpy as np
from scipy.io import arff
import requests
from tqdm import tqdm
import tsfel

from src.utils.pathtools import project
from src.utils.logs import logger

HEARTBEAT = 'AbnormalHeartbeat'
DIATOM = 'DiatomSizeReduction'
PRESSURE = 'PigAirwayPressure'
DATASETS = {
    name:f'https://timeseriesclassification.com/Downloads/{name}.zip'
    for name in [HEARTBEAT, DIATOM, PRESSURE]
}


class Dataset():

    def __init__(self, name:str) -> None:
        logger.info(f'Initiating dataset {name}')
        logger.info(f'More info at https://timeseriesclassification.com/description.php?Dataset={name}')
        self.name = name
        self._data = None
        self._features = None
        self._dir_path = project.data / self.name
        self._zip_path = project.data / f'{self.name}.zip'

    @property
    def data(self) -> pd.DataFrame:
        """Returns a pd.DataFrame of shape (m, T) 
        where m is the numpyer of time series and T the number of timestamp in the series."""
        if self._data is None:
            self._get_data()
        return self._data
    
    @property
    def features(self) -> np.ndarray:
        """Returns a np.ndarray of shape (n_features, m)
        where m is the number of time series"""
        if self._features is None:
            self._get_features()
        return self._features
    
    def __len__(self):
        return len(self.data.index)
    
# -------------------- GET DATA ---------------------

    def _get_data(self):
        """Gets data and load it in self._data"""
        self._check_downloaded()
        self._load_data()

    def _check_downloaded(self):
        """Checks that the data is correctly downloaded"""

        if not self._dir_path.exists() or len(list(self._dir_path.iterdir())) <= 2:
            logger.info(f'{self.name} dataset not found')

            if not self._zip_path.exists():
                logger.info(f'{self.name}  dataset zip not found, downloading it...')
                response = requests.get(DATASETS[self.name], stream=True)
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

            logger.info(f'Extracting {self.name}...')
            try:
                with zipfile.ZipFile(self._zip_path) as zf:
                    zf.extractall(project.mkdir_if_not_exists(self._dir_path))
            except zipfile.BadZipFile:
                logger.info(f'Found corrupted .zip file, deleting it and trying again...')
                self._zip_path.unlink()
                self._check_downloaded()

        else:
            logger.info(f'{self.name} found at {project.as_relative(self._dir_path)}')

    def _load_data(self):
        """Loads data in `self.data`"""
        logger.info(f'Loading dataset from disk: {self.name}')
        data_train_arff = arff.loadarff(self._dir_path / f'{self.name}_TRAIN.arff')
        data_train_df = pd.DataFrame(data_train_arff[0])
        data_test_arff = arff.loadarff(self._dir_path / f'{self.name}_TEST.arff')
        data_test_df = pd.DataFrame(data_test_arff[0])
        
        self._data = pd.concat(
            [data_train_df, data_test_df],
            ignore_index=True,
        )

# -------------------- GET DATA ---------------------

    def _get_features(self):
        """Extracts the features with tsfel"""
        logger.info(f'Extracting the features of {len(self)} time series')
        result = list()
        for index in tqdm(range(len(self))):
            serie = self.data.iloc[index, :-1]
            cfg = tsfel.get_features_by_domain()
            result.append(np.squeeze(
                tsfel.time_series_features_extractor(cfg, serie, verbose=0).values
            ))

        self._features = np.array(result).transpose()
        assert self._features.shape[1] == len(self), f'Incorrect features shape: {self._features.shape[1]} != {len(self)}'

heartbeat_ds = Dataset(HEARTBEAT)
diatom_ds = Dataset(DIATOM)
pressure_ds = Dataset(PRESSURE)