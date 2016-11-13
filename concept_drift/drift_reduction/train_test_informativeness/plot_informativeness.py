import click
import pandas as pd


from concept_drift.drift_reduction.train_test_informativeness.train_test_informativeness import get_drift_informativeness
from concept_drift.drift_reduction.utils import get_classification_informativeness, plot_informativeness
from concept_drift.score_calculator.score_calculation import get_labels_from_file


@click.command()
@click.argument('training-data-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('training-labels-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--info-measure-name', '-i', type=click.STRING, default='mutual_info')
def main(training_data_path, training_labels_path, test_data_path, info_measure_name):
    training_data = pd.read_csv(training_data_path, header=None, dtype='float32')
    training_labels = get_labels_from_file(training_labels_path)
    test_data = pd.read_csv(test_data_path, header=None, dtype='float32')
    x = get_classification_informativeness(
        training_data,
        training_labels,
        informativeness_measure_name=info_measure_name
    )
    y = get_drift_informativeness(
        training_data,
        test_data,
        informativeness_measure_name=info_measure_name
    )
    plot_informativeness(x, y)


if __name__ == '__main__':
    main()
