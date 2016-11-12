import csv
import os

from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

from settings import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, 'trees_drift_cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
CLASS_INFO_FILE = os.path.join(CACHE_DIR, 'classifiction_informativeness.csv')


def read_info_from_cache_file(cache_file_path):
    with open(cache_file_path, 'r') as cache_file:
        reader = csv.reader(cache_file)
        rows = list(reader)
        assert len(rows) == 1
        return rows[0]


def write_info_to_cache_file(cache_file_path, mutual_info):
    with open(cache_file_path, 'w') as cache_file:
        writer = csv.writer(cache_file)
        writer.writerow(mutual_info)


def get_classification_informativeness(X, y, use_cache=True):
    if use_cache and os.path.isfile(CLASS_INFO_FILE):
        return read_info_from_cache_file(CLASS_INFO_FILE)
    mutual_info = mutual_info_classif(X, y, discrete_features=False, copy=True, n_neighbors=10)
    write_info_to_cache_file(CLASS_INFO_FILE, mutual_info)


def plot_informativeness(classification_informativeness, drift_informativeness):
    plt.xlabel('classification informativeness')
    plt.ylabel('drift informativeness')
    plt.scatter(classification_informativeness, drift_informativeness)
