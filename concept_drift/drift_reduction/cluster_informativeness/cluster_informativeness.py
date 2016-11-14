import csv
from collections import defaultdict
import os

from sklearn.cluster import KMeans

from concept_drift.drift_reduction.feature_informativeness_measures import get_feature_informativeness_measure
from concept_drift.drift_reduction.utils import read_info_from_cache_file, write_info_to_cache_file, CACHE_DIR
from concept_drift.score_calculator.score_calculation import get_labels_from_file
from settings import SEED

CACHE_FILE_TEMPLATE = os.path.join(CACHE_DIR, 'cluster_informativeness_{}.csv')
CLUSTER_LABELS_FILE = os.path.join(CACHE_DIR, 'cluster_labels.csv')
# We have 4 firefighters
NUM_OF_CLUSTERS = 4


def write_labels_to_file(file_path, labels):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label])


def cluster(X):
    clustering = KMeans(
        n_clusters=NUM_OF_CLUSTERS,
        max_iter=1000,
        tol=0.0000001,
        verbose=0,
        copy_x=False,
        n_jobs=1,
        random_state=SEED
    )
    return clustering.fit_predict(X)


def group_by_label(y):
    indices_for_label = defaultdict(list)
    for index, label in enumerate(y):
        indices_for_label[label].append(index)
    return indices_for_label


def get_cluster_labels(X_for_clustering, y, use_cache=True):
    if use_cache and os.path.isfile(CLUSTER_LABELS_FILE):
        return get_labels_from_file(CLUSTER_LABELS_FILE)
    indices_for_label = group_by_label(y)
    labels_after_clustering = [0 for _ in xrange(len(y))]
    for base_label, indices in indices_for_label.iteritems():
        X_copy = X_for_clustering.iloc[indices, :]
        assert X_copy.shape == (len(indices), X_for_clustering.shape[1])
        cluster_labels = cluster(X_copy)
        for index, cluster_label in zip(indices, cluster_labels):
            final_label = '{}_{}'.format(base_label, cluster_label)
            labels_after_clustering[index] = final_label
    write_labels_to_file(CLUSTER_LABELS_FILE, labels_after_clustering)
    return labels_after_clustering


def get_cluster_drift_informativeness(X, X_for_clustering, y,
                                      informativeness_measure_name='mutual_info', use_cache=True):
    cache_file_name = CACHE_FILE_TEMPLATE.format(informativeness_measure_name)
    if use_cache and os.path.isfile(cache_file_name):
        return read_info_from_cache_file(cache_file_name)
    y_cluster = get_cluster_labels(X_for_clustering, y, use_cache=use_cache)
    informativeness_measure = get_feature_informativeness_measure(informativeness_measure_name)
    informativeness = informativeness_measure(X, y_cluster)
    write_info_to_cache_file(cache_file_name, informativeness)
    return informativeness
