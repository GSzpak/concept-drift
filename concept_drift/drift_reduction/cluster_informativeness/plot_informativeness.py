import click
import pandas as pd

from concept_drift.drift_reduction.cluster_informativeness.utils import VERSION_TO_INFO_CALC
from concept_drift.drift_reduction.utils import get_classification_informativeness, plot_informativeness
from concept_drift.score_calculator.score_calculation import get_labels_from_file


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data-for-clustering-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--version', '-v', type=click.INT, default=1)
@click.option('--info-measure-name', '-i', type=click.STRING, default='mutual_info')
def main(training_data_path, training_labels_path, data_for_clustering_path, version, info_measure_name):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    training_data_for_clustering = pd.read_csv(data_for_clustering_path, header=None, dtype='float32')
    x = get_classification_informativeness(
        training_data,
        training_labels,
        informativeness_measure_name=info_measure_name
    )
    info_calculator = VERSION_TO_INFO_CALC[version]
    y = info_calculator.get_cluster_drift_informativeness(
        training_data,
        training_data_for_clustering,
        training_labels,
        informativeness_measure_name=info_measure_name
    )
    plot_informativeness(x, y)


if __name__ == '__main__':
    main()
