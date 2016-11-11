import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from concept_drift.score_calculator.score_calculation import get_labels_from_file, balanced_accuracy


def balanced_accuracy_score(estimator, X, y):
    y_pred = estimator.predict(X)
    return balanced_accuracy(y, y_pred)


def calculate_train_score(clf, training_data, training_labels):
    scores = cross_val_score(clf, training_data, training_labels, cv=5, scoring=balanced_accuracy_score)
    return np.mean(scores)


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
    classifier = RandomForestClassifier(n_estimators=50, max_features=0.1, verbose=1, n_jobs=-1)
    train_score = calculate_train_score(classifier, training_data, training_labels)
    print 'Training set score (5 - fold CV): {}'.format(train_score)
    classifier.fit(training_data, training_labels)
    test_score = balanced_accuracy_score(classifier, test_data, test_labels)
    print 'Test set score: {}'.format(test_score)


if __name__ == '__main__':
    main()
