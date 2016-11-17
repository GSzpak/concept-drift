import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from concept_drift.classifier.perform_classification import ClassifierFactory, perform_classification
from concept_drift.drift_reduction.feature_informativeness_measures import get_feature_informativeness_measure
from settings import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, 'drift_reduction_cache_v2')
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
    return informativeness


def plot_informativeness(classification_informativeness, drift_informativeness):
    plt.xlabel('classification informativeness')
    plt.ylabel('drift informativeness')
    plt.scatter(classification_informativeness, drift_informativeness)
    plt.show()


def get_columns_to_drop(classif_info, drift_info, classif_lower_bound, drift_upper_bound):
    columns_to_drop = []
    for index, (classif_info, drift_info) in enumerate(zip(classif_info, drift_info)):
        if classif_info < classif_lower_bound or drift_info > drift_upper_bound \
                or np.isnan(classif_info) or np.isnan(drift_info):
            columns_to_drop.append(index)
    return columns_to_drop


def get_classification_report(training_data, training_labels, test_data, test_labels,
                              classif_info, drift_info, classif_lower_bound, drift_upper_bound):
    assert training_data.shape[1] == test_data.shape[1]
    num_of_columns = training_data.shape[1]
    print 'Calculating report for thresholds: {}, {}'.format(classif_lower_bound, drift_upper_bound)
    columns_to_drop = get_columns_to_drop(classif_info, drift_info, classif_lower_bound, drift_upper_bound)
    features_selected = num_of_columns - len(columns_to_drop)
    print '{} features selected'.format(features_selected)
    training_data_prepared = training_data.drop(training_data.columns[columns_to_drop], axis=1, inplace=False)
    test_data_prepared = test_data.drop(test_data.columns[columns_to_drop], axis=1, inplace=False)
    report_rows = []
    for classifier_name in ClassifierFactory.CLASSIFIERS:
        train_score, test_score = perform_classification(
            training_data_prepared,
            training_labels,
            test_data_prepared,
            test_labels,
            classifier_name
        )
        print '{}: train score {}, test score {}'.format(classifier_name, train_score, test_score)
        report_rows.append([classif_lower_bound, drift_upper_bound, classifier_name,
                            features_selected, '{0:.4f}'.format(train_score), '{0:.4f}'.format(test_score)])
    return report_rows
