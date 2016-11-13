import csv
import os

import matplotlib.pyplot as plt

from concept_drift.drift_reduction.feature_informativeness_measures import get_feature_informativeness_measure
from settings import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, 'drift_reduction_cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
CLASS_INFO_FILE_TEMPLATE = os.path.join(CACHE_DIR, 'classification_informativeness_{}.csv')


def read_info_from_cache_file(cache_file_path):
    with open(cache_file_path, 'r') as cache_file:
        reader = csv.reader(cache_file)
        rows = list(reader)
        assert len(rows) == 1
        result = rows[0]
        return map(float, result)


def write_info_to_cache_file(cache_file_path, feature_informativeness):
    with open(cache_file_path, 'w') as cache_file:
        writer = csv.writer(cache_file)
        writer.writerow(feature_informativeness)


def get_classification_informativeness(X, y, informativeness_measure_name='mutual_info', use_cache=True):
    cache_file_name = CLASS_INFO_FILE_TEMPLATE.format(informativeness_measure_name)
    if use_cache and os.path.isfile(cache_file_name):
        return read_info_from_cache_file(cache_file_name)
    informativeness_measure = get_feature_informativeness_measure(informativeness_measure_name)
    informativeness = informativeness_measure(X, y)
    write_info_to_cache_file(cache_file_name, informativeness)


def plot_informativeness(classification_informativeness, drift_informativeness):
    plt.xlabel('classification informativeness')
    plt.ylabel('drift informativeness')
    plt.scatter(classification_informativeness, drift_informativeness)
    plt.show()

