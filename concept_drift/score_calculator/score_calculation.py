import csv
from collections import Counter, defaultdict

import itertools

import click


def balanced_accuracy(y_true, y_pred):
    labels_counts = Counter(y_true)
    label_accuracy = defaultdict(int)
    for label_pred, label_act in itertools.izip(y_pred, y_true):
        if label_pred == label_act:
            label_accuracy[label_act] += 1
    accuracy_sum = sum([label_accuracy[label] / float(label_count)
                        for label, label_count in labels_counts.iteritems()])
    return accuracy_sum / float(len(labels_counts))


def calculate_score(postures, postures_act, activities, activities_act):
    posture_bac = balanced_accuracy(postures, postures_act)
    activity_bac = balanced_accuracy(activities, activities_act)
    print 'posture BAC: {}, activity BAC: {}'.format(posture_bac, activity_bac)
    return (posture_bac + 2 * activity_bac) / 3.


def get_labels_from_file(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]


def get_score_from_file(predictions_file_path, labels_file_path):
    activities = get_labels_from_file(predictions_file_path)
    activities_true = get_labels_from_file(labels_file_path)
    return balanced_accuracy(activities_true, activities)


@click.command()
@click.argument('predictions-file-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('labels-file-path', type=click.Path(exists=True, dir_okay=False))
def main(predictions_file_path, labels_file_path):
    print get_score_from_file(predictions_file_path, labels_file_path)


if __name__ == '__main__':
    main()

# TODO: calculate final score