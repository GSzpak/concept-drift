import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from concept_drift.score_calculator.score_calculation import get_labels_from_file, balanced_accuracy


class ClassifierFactory(object):

    SEED = 500
    NAME_TO_CLF = {
        'random_forest': (
            RandomForestClassifier,
            dict(n_estimators=50, max_features=0.1, random_state=SEED, verbose=1, n_jobs=-1)
        ),
        'logit': (
            SGDClassifier,
            dict(loss='log', alpha=0.0001, n_iter=100, random_state=SEED, verbose=0, n_jobs=-1)
        ),
        'svm': (
            SGDClassifier,
            dict(loss='hinge', alpha=0.0001, n_iter=100, random_state=SEED, verbose=0, n_jobs=-1)
        ),
    }
    CLASSIFIERS = NAME_TO_CLF.keys()

    @staticmethod
    def make_classifier(clf_name):
        if clf_name not in ClassifierFactory.NAME_TO_CLF:
            raise ValueError('Unknown classifier: {}'.format(clf_name))
        clf_class, clf_params = ClassifierFactory.NAME_TO_CLF[clf_name]
        return clf_class(**clf_params)


def balanced_accuracy_score(estimator, X, y):
    y_pred = estimator.predict(X)
    return balanced_accuracy(y, y_pred)


def calculate_train_score(clf, training_data, training_labels):
    scores = cross_val_score(clf, training_data, training_labels, cv=3, scoring=balanced_accuracy_score)
    return np.mean(scores)


def perform_classification(training_data, training_labels, test_data, test_labels, classifier_name):
    classifier = ClassifierFactory.make_classifier(classifier_name)
    train_score = calculate_train_score(classifier, training_data, training_labels)
    classifier.fit(training_data, training_labels)
    test_score = balanced_accuracy_score(classifier, test_data, test_labels)
    return train_score, test_score


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--classifier-name', '-c', type=click.STRING, default='random_forest')
def main(training_data_path, training_labels_path, test_data_path, test_labels_path, classifier_name):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    test_data = pd.read_csv(test_data_path, header=None, dtype='float32')
    test_labels = get_labels_from_file(test_labels_path)
    train_score, test_score = perform_classification(
        training_data,
        training_labels,
        test_data,
        test_labels,
        classifier_name
    )
    print 'Training set score (3 - fold CV): {}'.format(train_score)
    print 'Test set score: {}'.format(test_score)


if __name__ == '__main__':
    main()
