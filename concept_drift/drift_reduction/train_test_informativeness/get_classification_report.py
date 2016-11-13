import csv

import click
import pandas as pd


# TODO: fill this
from concept_drift.classifier.perform_classification import ClassifierFactory, perform_classification
from concept_drift.drift_reduction.train_test_informativeness.train_test_informativeness import \
    get_drift_informativeness
from concept_drift.drift_reduction.utils import get_classification_informativeness
from concept_drift.score_calculator.score_calculation import get_labels_from_file

# First - classification lower bound
# Second - drift upper bound
THRESHOLD_PAIRS_TO_TEST = []


def get_columns_to_drop(classif_info, drift_info, classif_lower_bound, drift_upper_bound):
    columns_to_drop = []
    for index, (classif_info, drift_info) in enumerate(zip(classif_info, drift_info)):
        if classif_info < classif_lower_bound or drift_info > drift_upper_bound:
            columns_to_drop.append(index)
    return columns_to_drop


def get_classification_report(training_data, training_labels, test_data, test_labels,
                              classif_info, drift_info, classif_lower_bound, drift_upper_bound):
    columns_to_drop = get_columns_to_drop(classif_info, drift_info, classif_lower_bound, drift_upper_bound)
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
        report_rows.append([classif_lower_bound, drift_upper_bound, classifier_name, train_score, test_score])
    return report_rows


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('report-file-path', type=click.Path(exists=False, dir_okay=False))
def main(training_data_path, training_labels_path, test_data_path, test_labels_path, report_file_path):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    test_data = pd.read_csv(test_data_path, header=None, dtype='float32')
    test_labels = get_labels_from_file(test_labels_path)
    classif_info = get_classification_informativeness(training_data, training_labels)
    drift_info = get_drift_informativeness(training_data, test_data)
    with open(report_file_path, 'w') as report_file:
        report_writer = csv.writer(report_file)
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
