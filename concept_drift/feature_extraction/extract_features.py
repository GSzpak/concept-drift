import collections
import csv
import glob
import itertools
import os

import click
import numpy as np

from concept_drift.data_preparation.prepare_data import BASE_FEATURES_FILE_NAME, TIME_FILE_NAME, SERIES_DIR_NAME, \
    NUM_OF_SERIES
from concept_drift.feature_extraction.cross_series_features import cross_correlation
from concept_drift.feature_extraction.physical_features import PHYSICAL_FEATURES
from concept_drift.feature_extraction.statistical_features import STATISTICAL_FEATURES
from concept_drift.feature_extraction.time_related_features import get_time_related_features

FEATURE_EXTRACTION_FUNS = STATISTICAL_FEATURES + PHYSICAL_FEATURES


def make_series_files(series_dir):
    series_filenames = glob.glob(os.path.join(series_dir, '*.csv'))
    # all series minus time measurments
    assert len(series_filenames) == NUM_OF_SERIES - 1
    return [open(series_filename, 'r') for series_filename in series_filenames]


def get_next_timeseries_list(series_readers):
    series_list = [next(reader) for reader in series_readers]
    return [np.asarray(time_series, dtype=np.float32) for time_series in series_list]


def flatten(iterable):
    result = []
    for elem in iterable:
        if isinstance(elem, collections.Iterable):
            result.extend(flatten(elem))
        else:
            result.append(elem)
    return result


def get_features(time_series_list, time):
    result = []
    for time_series in time_series_list:
        features = [fun(time_series) for fun in FEATURE_EXTRACTION_FUNS]
        result.extend(flatten(features))
        time_related_features = get_time_related_features(time_series, time)
        result.extend(flatten(time_related_features))
    for i in xrange(len(time_series_list)):
        for j in xrange(i + 1, len(time_series_list)):
            result.append(cross_correlation(time_series_list[i], time_series_list[j]))
    return result


@click.command()
@click.argument('data-dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('out-file-path', type=click.Path(exists=False, dir_okay=False))
def extract_features(data_dir, out_file_path):
    base_features_file_path = os.path.join(data_dir, BASE_FEATURES_FILE_NAME)
    time_file_path = os.path.join(data_dir, TIME_FILE_NAME)
    series_dir = os.path.join(data_dir, SERIES_DIR_NAME)
    with open(base_features_file_path, 'r') as base_features_file, \
            open(time_file_path, 'r') as time_file, \
            open(out_file_path, 'w') as out_file:
        series_files = make_series_files(series_dir)
        series_readers = [csv.reader(series_file) for series_file in series_files]
        base_features_reader = csv.reader(base_features_file)
        time_reader = csv.reader(time_file)
        out_writer = csv.writer(out_file)
        for progress, (base_features, time) in enumerate(itertools.izip(base_features_reader, time_reader)):
            features = base_features
            # Unused so far
            time = np.asarray(time, dtype=np.float32)
            time_series_list = get_next_timeseries_list(series_readers)
            time_series_features = get_features(time_series_list, time)
            features.extend(time_series_features)
            out_writer.writerow(features)
            if progress % 100 == 0:
                print 'Progress: {}'.format(progress)
        for series_file in series_files:
            series_file.close()


if __name__ == '__main__':
    extract_features()
