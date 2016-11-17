import csv
from collections import defaultdict
import os

import numpy as np
from sklearn.cluster import KMeans

from concept_drift.drift_reduction.feature_informativeness_measures import get_feature_informativeness_measure
from concept_drift.drift_reduction.utils import read_info_from_cache_file, write_info_to_cache_file, CACHE_DIR
from concept_drift.score_calculator.score_calculation import get_labels_from_file
from settings import SEED


class ClusterInformativenessCalculator(object):

    # We have 4 firefighters
    NUM_OF_CLUSTERS = 4

    @staticmethod
    def group_by_label(y):
        indices_for_label = defaultdict(list)
        for index, label in enumerate(y):
            indices_for_label[label].append(index)
        return indices_for_label

    def make_cache_file_name(self, measure_name):
        raise NotImplementedError

    def cluster(self, X):
        clustering = KMeans(
            n_clusters=self.NUM_OF_CLUSTERS,
            max_iter=1000,
            n_init=20,
            tol=0.0000001,
            verbose=0,
            copy_x=False,
            n_jobs=1,
            random_state=SEED
        )
        return clustering.fit_predict(X)

    def _get_cluster_labels(self, X, indices):
        X_copy = X.iloc[indices, :]
        assert X_copy.shape == (len(indices), X.shape[1])
        return self.cluster(X_copy)

    def _calculate_drift_informativeness(self, X, X_for_clustering, indices_for_label, informativeness_measure):
        raise NotImplementedError

    def get_cluster_drift_informativeness(self, X, X_for_clustering, y,
                                          informativeness_measure_name='mutual_info', use_cache=True):
        cache_file_name = self.make_cache_file_name(informativeness_measure_name)
        if use_cache and os.path.isfile(cache_file_name):
            return read_info_from_cache_file(cache_file_name)
        informativeness_measure = get_feature_informativeness_measure(informativeness_measure_name)
        indices_for_label = self.group_by_label(y)
        informativeness = self._calculate_drift_informativeness(
            X,
            X_for_clustering,
            indices_for_label,
            informativeness_measure
        )
        write_info_to_cache_file(cache_file_name, informativeness)
        return informativeness


class ClusterInformativenessCalculatorV1(ClusterInformativenessCalculator):

    _CACHE_FILE_TEMPLATE = os.path.join(CACHE_DIR, 'cluster_informativeness_v1_{}.csv')
    _CLUSTER_LABELS_FILE = os.path.join(CACHE_DIR, 'cluster_labels.csv')

    @staticmethod
    def write_labels_to_file(file_path, labels):
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            for label in labels:
                writer.writerow([label])

    def make_cache_file_name(self, measure_name):
        return self._CACHE_FILE_TEMPLATE.format(measure_name)

    def _get_clustered_labels(self, X_for_clustering, indices_for_label):
        if os.path.isfile(self._CLUSTER_LABELS_FILE):
            return get_labels_from_file(self._CLUSTER_LABELS_FILE)
        num_of_instances = sum([len(indices) for indices in indices_for_label.values()])
        labels_after_clustering = [0 for _ in xrange(num_of_instances)]
        for base_label, indices in indices_for_label.iteritems():
            cluster_labels = self._get_cluster_labels(X_for_clustering, indices)
            for index, cluster_label in zip(indices, cluster_labels):
                final_label = '{}_{}'.format(base_label, cluster_label)
                labels_after_clustering[index] = final_label
        self.write_labels_to_file(self._CLUSTER_LABELS_FILE, labels_after_clustering)
        return labels_after_clustering

    def _calculate_drift_informativeness(self, X, X_for_clustering, indices_for_label, informativeness_measure):
        y_cluster = self._get_clustered_labels(X_for_clustering, indices_for_label)
        return informativeness_measure(X, y_cluster)


class ClusterInformativenessCalculatorV2(ClusterInformativenessCalculator):

    _CACHE_FILE_TEMPLATE = os.path.join(CACHE_DIR, 'cluster_informativeness_v2_{}.csv')
    _PARTIAL_RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, 'cluster_v2_partial_results')
    if not os.path.exists(_PARTIAL_RESULTS_CACHE_DIR):
        os.makedirs(_PARTIAL_RESULTS_CACHE_DIR)

    def make_cache_file_name(self, measure_name):
        return self._CACHE_FILE_TEMPLATE.format(measure_name)

    def _make_partial_results_cache_name(self, label, measure_name):
        return os.path.join(self._PARTIAL_RESULTS_CACHE_DIR, '{}_{}.csv'.format(label, measure_name))

    def _get_informativeness_for_label(self, X, X_for_clustering, label, indices, informativeness_measure):
        cache_file_name = self._make_partial_results_cache_name(label, informativeness_measure.__name__)
        if os.path.isfile(cache_file_name):
            return read_info_from_cache_file(cache_file_name)
        cluster_labels = self._get_cluster_labels(X_for_clustering, indices)
        X_part = X.iloc[indices, :]
        label_informativeness = informativeness_measure(X_part, cluster_labels)
        write_info_to_cache_file(cache_file_name, label_informativeness)
        return label_informativeness

    def _calculate_drift_informativeness(self, X, X_for_clustering, indices_for_label, informativeness_measure):
        results = []
        for label, indices in indices_for_label.iteritems():
            print 'Calculating informativeness for label {}, {} instances'.format(label, len(indices))
            partial_informativeness = self._get_informativeness_for_label(
                X,
                X_for_clustering,
                label,
                indices,
                informativeness_measure
            )
            results.append(partial_informativeness)
        assert len(results) == len(indices_for_label)
        result = np.mean(results, axis=0)
        assert len(result) == X.shape[1]
        return result
