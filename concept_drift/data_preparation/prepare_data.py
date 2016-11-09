import csv
import os

import click
import itertools

from settings import DATA_DIR


OUT_DIR = os.path.join(DATA_DIR, 'prepared')
SERIES_DIR = os.path.join(OUT_DIR, 'series')
NUM_OF_FEATURES = 17242
NUM_OF_SERIES = 43
NUM_OF_ESSK_FEATURES = 42
SERIES_LEN = 400


def extract_essk_features(row):
    return row[:NUM_OF_ESSK_FEATURES]


def extract_series(row, start_index):
    result = row[start_index::NUM_OF_SERIES]
    assert len(result) == 400
    return result


def split_data_to_series(row):
    essk_features = extract_essk_features(row)
    time_measurments = extract_series(row, NUM_OF_ESSK_FEATURES)
    series = [extract_series(row, start_index)
              for start_index in xrange(NUM_OF_ESSK_FEATURES + 1, NUM_OF_ESSK_FEATURES + NUM_OF_SERIES)]
    assert len(essk_features) + len(time_measurments) + sum([len(s) for s in series]) == NUM_OF_FEATURES
    return essk_features, time_measurments, series


def open_series_files():
    series_filenames = [os.path.join(SERIES_DIR, '{}.csv'.format(i)) for i in xrange(1, NUM_OF_SERIES)]
    return [open(filename, 'w') for filename in series_filenames]


@click.command()
@click.argument('training-data-file-path', type=click.Path(exists=True, dir_okay=False))
def main(training_data_file_path):
    essk_file_path = os.path.join(OUT_DIR, 'essk.csv')
    time_file_path = os.path.join(OUT_DIR, 'time.csv')
    with open(training_data_file_path, 'r') as training_data_file, \
            open(essk_file_path, 'w') as essk_file, \
            open(time_file_path, 'w') as time_file:
        series_files = open_series_files()
        training_data_reader = csv.reader(training_data_file)
        essk_writer = csv.writer(essk_file)
        time_writer = csv.writer(time_file)
        series_writers = [csv.writer(series_file) for series_file in series_files]
        for progress, row in enumerate(training_data_reader):
            essk_features, time_measurments, series = split_data_to_series(row)
            essk_writer.writerow(essk_features)
            time_writer.writerow(time_measurments)
            for series_writer, series_row in itertools.izip(series_writers, series):
                series_writer.writerow(series_row)
            if progress % 1000 == 0:
                print 'Progress: {}'.format(progress)
        for series_file in series_files:
            series_file.close()


if __name__ == '__main__':
    main()
