import csv
import os

import click
import pandas as pd

from concept_drift.classifier.perform_classification import ClassifierFactory, get_classification_scores
from concept_drift.iterative_learning.classifier import IterativeClassifier
from concept_drift.score_calculator.score_calculation import get_labels_from_file
from settings import THESIS_DATA_DIR

PERCENTAGES_TO_TEST = [0.1, 0.25, 0.34, 0.5]


def _make_report_file_path(clf_name):
    return os.path.join(THESIS_DATA_DIR, 'iterative_{}_report.csv'.format(clf_name))


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
    for classifier_name in ClassifierFactory.CLASSIFIERS:
        report_file_path = _make_report_file_path(classifier_name)
        with open(report_file_path, 'w') as report_file:
                report_writer = csv.writer(report_file)
                report_writer.writerow([
                    'rows percentage',
                    'train score',
                    'test score'
                ])
                for rows_percentage in PERCENTAGES_TO_TEST:
                    base_classifier = ClassifierFactory.make_classifier(classifier_name)
                    classifier = IterativeClassifier(
                        base_classifier=base_classifier,
                        rows_percentage=rows_percentage
                    )
                    train_score, test_score = get_classification_scores(
                        training_data,
                        training_labels,
                        test_data,
                        test_labels,
                        classifier
                    )
                    report_writer.writerow([rows_percentage, train_score, test_score])


if __name__ == '__main__':
    main()
