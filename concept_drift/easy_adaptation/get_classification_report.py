import csv
import os

import click
import numpy as np
import pandas as pd

from concept_drift.classifier.perform_classification import ClassifierFactory, perform_classification
from concept_drift.drift_reduction.train_test_informativeness.train_test_informativeness import \
    get_drift_informativeness
from concept_drift.drift_reduction.utils import get_classification_informativeness
from concept_drift.score_calculator.score_calculation import get_labels_from_file

from settings import THESIS_DATA_DIR
REPORT_PATH = os.path.join(THESIS_DATA_DIR, 'easy_report.csv')


# First - classification lower bound
# Second - drift upper bound
THRESHOLD_PAIRS_TO_TEST = [
    (0, 8000), (250, 250), (250, 500),
    (500, 250), (500, 500),
]


def get_columns_to_drop(classif_info, drift_info, classif_lower_bound, drift_upper_bound):
    columns_to_drop = []
    for index, (classif_info, drift_info) in enumerate(zip(classif_info, drift_info)):
        if classif_info < classif_lower_bound or drift_info > drift_upper_bound:
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
    zeros_train = pd.DataFrame(np.zeros((training_data.shape[0], features_selected), dtype=np.float32))
    zeros_test = pd.DataFrame(np.zeros((test_data.shape[0], features_selected), dtype=np.float32))
    training_data_prepared = training_data.drop(training_data.columns[columns_to_drop], axis=1, inplace=False)
    training_data_adapted = pd.concat(
        [training_data_prepared,
         training_data_prepared.copy(),
         zeros_train],
        axis=1,
        ignore_index=True
    )
    test_data_prepared = test_data.drop(test_data.columns[columns_to_drop], axis=1, inplace=False)
    test_data_adapted = pd.concat(
        [test_data_prepared,
         zeros_test,
         test_data_prepared.copy()],
        axis=1,
        ignore_index=True
    )
    assert training_data_adapted.shape == (training_data.shape[0], 3 * features_selected)
    assert test_data_adapted.shape == (test_data.shape[0], 3 * features_selected)
    report_rows = []
    for classifier_name in ClassifierFactory.CLASSIFIERS:
        train_score, test_score = perform_classification(
            training_data_adapted,
            training_labels,
            test_data_adapted,
            test_labels,
            classifier_name
        )
        print '{}: train score {}, test score {}'.format(classifier_name, train_score, test_score)
        report_rows.append([classifier_name, features_selected, '{0:.4f}'.format(train_score), '{0:.4f}'.format(test_score)])
    return report_rows


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-labels-path', type=click.Path(exists=True, dir_okay=False))
def main(training_data_path, training_labels_path, test_data_path, test_labels_path):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    test_data = pd.read_csv(test_data_path, header=None, dtype='float32')
    test_labels = get_labels_from_file(test_labels_path)
    classif_info = get_classification_informativeness(
        training_data,
        training_labels,
        informativeness_measure_name='anova_f'
    )
    drift_info = get_drift_informativeness(
        training_data,
        test_data,
        informativeness_measure_name='anova_f'
    )
    with open(REPORT_PATH, 'w') as report_file:
        report_writer = csv.writer(report_file)
        report_writer.writerow([
            'classifier name',
            'number of features',
            'train score',
            'test score'
        ])
        for classif_lower_bound, drift_upper_bound in THRESHOLD_PAIRS_TO_TEST:
            report_rows = get_classification_report(
                training_data,
                training_labels,
                test_data,
                test_labels,
                classif_info,
                drift_info,
                classif_lower_bound,
                drift_upper_bound
            )
            report_writer.writerows(report_rows)


if __name__ == '__main__':
    main()
