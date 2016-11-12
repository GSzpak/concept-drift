import os

from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import shuffle

from concept_drift.drift_reduction.utils import read_info_from_cache_file, write_info_to_cache_file, CACHE_DIR

DRIFT_INFO_FILE = os.path.join(CACHE_DIR, 'train_test_informativeness.csv')


def get_drift_informativeness(X_training, X_test, use_cache=True):
    if use_cache and os.path.isfile(DRIFT_INFO_FILE):
        return read_info_from_cache_file(DRIFT_INFO_FILE)
    num_training = X_training.shape[0]
    num_test = X_test.shape[0]
    y = ['training'] * num_training + ['test'] * num_test
    print X_training.shape, X_test.shape
    X = X_training.append(X_test, ignore_index=True)
    print X.shape
    X_shuffled, y_shuffled = shuffle(X, y)
    mutual_info = mutual_info_classif(X_shuffled, y_shuffled, discrete_features=False, copy=True, n_neighbors=10)
    write_info_to_cache_file(DRIFT_INFO_FILE, mutual_info)
