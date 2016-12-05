import csv
import glob
import os


def split_report(file_path):
    basename, ext = os.path.splitext(file_path)
    clf_names = ['logit', 'svm', 'random_forest']
    clf_to_file = {
        clf: open('{}_{}{}'.format(basename, clf, ext), 'w') for clf in clf_names
    }
    clf_to_writer = {
        clf: csv.writer(f) for clf, f in clf_to_file.iteritems()
    }
    with open(file_path, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        header_out = ['$threshold_{class}$', '$threshold_{drift}$', 'Liczba cech', '$X_{train}$ $BAC$', '$X_{test}$ $BAC$']
        for writer in clf_to_writer.values():
            writer.writerow(header_out)
        for row in reader:
            t_class, t_drift, clf, features, train, test = row
            clf_to_writer[clf].writerow([t_class, t_drift, features, train, test])
    for f in clf_to_file.values():
        f.close()


files = glob.glob('*.csv')
for file_path in files:
    if 'iterative' not in file_path:
        split_report(file_path)
