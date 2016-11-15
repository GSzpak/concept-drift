import csv

import click
import pandas as pd

from concept_drift.drift_reduction.train_test_informativeness.train_test_informativeness import \
    get_drift_informativeness
from concept_drift.drift_reduction.utils import get_classification_informativeness, get_classification_report
from concept_drift.score_calculator.score_calculation import get_labels_from_file

# First - classification lower bound
# Second - drift upper bound
THRESHOLD_PAIRS_TO_TEST = {
    'mutual_info': [
        (0.0, 0.7), (0.0, 0.05), (0.0, 0.1), (0.0, 0.2),
        (0.25, 0.05), (0.25, 0.1), (0.25, 0.2),
        (0.5, 0.05), (0.5, 0.1), (0.5, 0.2), (0.5, 0.4),
        (1.0, 0.1), (1.0, 0.3), (1.0, 0.5),
        (1.5, 0.7)
    ],
    'random_forest': [
        (0.0, 0.05), (0.0, 0.00001), (0.0, 0.00005), (0.0, 0.0001), (0.0, 0.001), (0.0, 0.0025), (0.0, 0.005),
        (0.00005, 0.00001), (0.00005, 0.00005), (0.00005, 0.0001), (0.00005, 0.001), (0.00005, 0.0025), (0.00005, 0.005),
        (0.0001, 0.00001), (0.0001, 0.00005), (0.0001, 0.0001), (0.0001, 0.001), (0.0001, 0.005),
        (0.0005, 0.00001), (0.0005, 0.00005), (0.0005, 0.0001), (0.0005, 0.001), (0.0005, 0.005), (0.0005, 0.05),
        (0.001, 0.0001), (0.001, 0.001), (0.001, 0.0025), (0.001, 0.005), (0.001, 0.05),
        (0.002, 0.05)
    ],
    'anova_f': [
        (0.0, 8000), (0.0, 50), (0.0, 100), (0.0, 250), (0.0, 500), (0.0, 750), (0.0, 1000), (0.0, 2000), (0.0, 4000),
        (250, 100), (250, 250), (250, 500), (250, 750), (250, 1000), (250, 2000),
        (500, 100), (500, 250), (500, 500), (500, 750), (500, 1000), (500, 2000),
        (750, 100), (750, 250), (750, 500), (750, 750), (750, 1000), (750, 2000),
        (1000, 100), (1000, 250), (1000, 500), (1000, 750), (1000, 1000), (1000, 2000),
        (2000, 1000), (2000, 2000)
    ]
}


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('report-file-path', type=click.Path(exists=False, dir_okay=False))
@click.option('--info-measure-name', '-i', type=click.STRING, default='mutual_info')
def main(training_data_path, training_labels_path, test_data_path, test_labels_path, report_file_path, info_measure_name):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    test_data = pd.read_csv(test_data_path, header=None, dtype='float32')
    test_labels = get_labels_from_file(test_labels_path)
    classif_info = get_classification_informativeness(
        training_data,
        training_labels,
        informativeness_measure_name=info_measure_name
    )
    drift_info = get_drift_informativeness(
        training_data,
        test_data,
        informativeness_measure_name=info_measure_name
    )
    pairs_to_test = THRESHOLD_PAIRS_TO_TEST[info_measure_name]
    with open(report_file_path, 'w') as report_file:
        report_writer = csv.writer(report_file)
        report_writer.writerow([
            'classification lower bound',
            'drift upper bound',
            'classifier name',
            'number of features',
            'train score',
            'test score'
        ])
        for classif_lower_bound, drift_upper_bound in pairs_to_test:
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
