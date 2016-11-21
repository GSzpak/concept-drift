import click
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier

from concept_drift.domain_adaptive_classification.classifier import BaseDomainAdaptiveClassifier
from concept_drift.score_calculator.score_calculation import get_labels_from_file, balanced_accuracy


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
    clf_base = BaseDomainAdaptiveClassifier(num_of_features=10)
    classifier = OneVsRestClassifier(clf_base)
    classifier.fit(training_data, training_labels)
    y_pred = classifier.predict(test_data)
    print balanced_accuracy(test_labels, y_pred)


if __name__ == '__main__':
    main()
