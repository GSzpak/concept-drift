import os

from sklearn.utils import shuffle

from concept_drift.drift_reduction.feature_informativeness_measures import get_feature_informativeness_measure
from concept_drift.drift_reduction.utils import read_info_from_cache_file, write_info_to_cache_file, CACHE_DIR

DRIFT_INFO_FILE_TEMPLATE = os.path.join(CACHE_DIR, 'train_test_informativeness_{}.csv')


def get_drift_informativeness(X_training, X_test, informativeness_measure_name='mutual_info', use_cache=True):
    cache_file_name = DRIFT_INFO_FILE_TEMPLATE.format(informativeness_measure_name)
    if use_cache and os.path.isfile(cache_file_name):
        return read_info_from_cache_file(cache_file_name)
    num_training = X_training.shape[0]
    num_test = X_test.shape[0]
    y = ['training'] * num_training + ['test'] * num_test
    X = X_training.append(X_test, ignore_index=True)
    X_shuffled, y_shuffled = shuffle(X, y)
    informativeness_measure = get_feature_informativeness_measure(informativeness_measure_name)
    informativeness = informativeness_measure(X_shuffled, y_shuffled)
    write_info_to_cache_file(cache_file_name, informativeness)
    return informativeness
