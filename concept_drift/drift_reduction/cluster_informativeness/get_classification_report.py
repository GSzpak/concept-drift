import csv
import os

import click
import pandas as pd

from concept_drift.drift_reduction.cluster_informativeness.utils import VERSION_TO_INFO_CALC
from concept_drift.drift_reduction.utils import get_classification_informativeness, get_classification_report
from concept_drift.score_calculator.score_calculation import get_labels_from_file
from settings import THESIS_DATA_DIR

# First - classification lower bound
# Second - drift upper bound
THRESHOLD_PAIRS_TO_TEST = {
    1: {
        'random_forest': [
            (0.0, 0.03), (0.0, 0.0005), (0.0, 0.00025), (0.0, 0.0001), (0.0, 0.00001),
            (0.00001, 0.0005), (0.00001, 0.00025), (0.00001, 0.0001), (0.00001, 0.00001),
            (0.00005, 0.0005), (0.00005, 0.00025), (0.00005, 0.0001), (0.00005, 0.00001),
            (0.0001, 0.0001), (0.0001, 0.0005), (0.0001, 0.00025), (0.0001, 0.0001),
            (0.0005, 0.0001), (0.0005, 0.0005), (0.0005, 0.00025),
            (0.001, 0.03)
        ],
    },
    2: {
        'mutual_info': [
            (0.0, 1.2), (0.0, 0.2), (0.0, 0.4), (0.0, 0.8),
            (0.5, 0.6), (0.5, 0.8),
            (1.0, 0.8), (1.0, 1.2),
            (1.5, 0.8), (1.5, 1.2),
        ],
        'random_forest': [
            (0.0, 0.4), (0.0, 0.002), (0.0, 0.0005), (0.0, 0.00005), (0.0, 2.5e-05), (0.0, 1.25e-05), (0.0, 1.0e-05),
            (2.5e-05, 0.4), (2.5e-05, 0.002), (2.5e-05, 0.0005), (2.5e-05, 0.00005), (2.5e-05, 2.5e-05), (2.5e-05, 1.25e-05),
            (0.00005, 0.4), (0.00005, 0.002), (0.00005, 0.0005), (0.00005, 0.00005), (0.00005, 2.5e-05), (0.00005, 1.25e-05),
            (0.00025, 0.4), (0.00025, 0.002), (0.00025, 0.0005), (0.00025, 0.00005), (0.00025, 2.5e-05),
            (0.0005, 0.4), (0.0005, 0.002), (0.0005, 0.0005), (0.0005, 0.00005), (0.0005, 2.5e-05),
            (0.001, 0.4), (0.001, 0.002), (0.001, 0.0005), (0.001, 0.00005)
        ],
        'anova_f': [
            (0.0, 1.4e7), (0.0, 10), (0.0, 50), (0.0, 100), (0.0, 250), (0.0, 500), (0.0, 2000),
            (500, 10), (500, 50), (500, 100), (500, 250), (500, 500), (500, 2000),
            (1000, 10), (1000, 50), (1000, 100), (1000, 250), (1000, 500), (1000, 2000),
            (2000, 10), (2000, 50), (2000, 100), (2000, 250), (2000, 500), (2000, 2000), (2000, 1.4e7),
            (3000, 1.4e7),
            (4000, 1.4e7),
        ]
    },
}


def _make_report_path(version, info_measure_name):
    return os.path.join(THESIS_DATA_DIR, 'clustering_v{}_{}_report.csv'.format(version, info_measure_name))


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data-for-clustering-path', type=click.Path(exists=True, dir_okay=False))
def main(training_data_path, training_labels_path, test_data_path, test_labels_path, data_for_clustering_path):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    test_data = pd.read_csv(test_data_path, header=None, dtype='float32')
    test_labels = get_labels_from_file(test_labels_path)
    data_for_clustering = pd.read_csv(data_for_clustering_path, header=None, dtype='float32')
    for version, measure_name_to_thresholds in THRESHOLD_PAIRS_TO_TEST.iteritems():
        info_calculator = VERSION_TO_INFO_CALC[version]
        for measure_name, threshold_pairs_to_test in measure_name_to_thresholds.iteritems():
            print 'Version: {}, measure: {}'.format(version, measure_name)
            classif_info = get_classification_informativeness(
                training_data,
                training_labels,
                informativeness_measure_name=measure_name
            )
            drift_info = info_calculator.get_cluster_drift_informativeness(
                training_data,
                data_for_clustering,
                training_labels,
                informativeness_measure_name=measure_name
            )
            report_file_path = _make_report_path(version, measure_name)
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
                for classif_lower_bound, drift_upper_bound in threshold_pairs_to_test:
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
