from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

from settings import SEED


def binarize(X):
    return (1 + np.sign(X)) / 2


def get_svm():
    return SGDClassifier(
        loss='hinge',
        alpha=0.0001,
        n_iter=3,
        random_state=SEED,
        verbose=0,
        n_jobs=-1
    )


def d(B, y):
    ones = B[[ind for ind, label in enumerate(y) if label == 1], :]
    minus_ones = B[[ind for ind, label in enumerate(y) if label == -1], :]
    result = 0
    for i in xrange(len(ones)):
        for j in xrange(i + 1, len(ones)):
            row1 = ones[i]
            row2 = ones[j]
            result += sum(abs(row1 - row2))
    for i in xrange(len(minus_ones)):
        for j in xrange(i + 1, len(minus_ones)):
            row1 = minus_ones[i]
            row2 = minus_ones[j]
            result += np.sum(np.abs(row1 - row2))
    for row1 in ones:
        for row2 in minus_ones:
            result -= sum(abs(row1 - row2))
    return result


def optimize_binary_codes(codes_matrix, labels):
    label_to_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)
    current_codes_matrix_T = codes_matrix.transpose()
    prev_codes_matrix_T = np.zeros(current_codes_matrix_T.shape)
    iter = 0
    while not (current_codes_matrix_T == prev_codes_matrix_T).all() and iter < 10:
        prev_codes_matrix_T = np.copy(current_codes_matrix_T)
        for code_bit_num, column in enumerate(prev_codes_matrix_T):
            num_equal_zero = sum([not bit for bit in column])
            num_equal_one = sum([bit for bit in column])
            for label, indices in label_to_indices.iteritems():
                num_equal_zero_in_class = sum([not column[i] for i in indices])
                num_equal_one_in_class = sum([column[i] for i in indices])
                num_equal_zero_outside_class = num_equal_zero - num_equal_zero_in_class
                num_equal_one_outside_class = num_equal_one - num_equal_one_in_class
                gradient_if_equal_zero = -(num_equal_one_in_class - num_equal_one_outside_class)
                gradient_if_equal_one = num_equal_zero_in_class - num_equal_zero_outside_class
                for i in indices:
                    if column[i]:
                        current_codes_matrix_T[code_bit_num, i] = int(1 - gradient_if_equal_one > 0)
                    else:
                        current_codes_matrix_T[code_bit_num, i] = int(-gradient_if_equal_zero > 0)
        iter += 1
    return current_codes_matrix_T.transpose()


def get_discr_binary_codes(X, y, num_of_features):
    pca = PCA(n_components=num_of_features)
    svm_classifier = get_svm()
    B = pca.fit_transform(X)
    B = binarize(B)
    hyperplanes_T = np.zeros((num_of_features, X.shape[1]))
    while True:
        B_prim = optimize_binary_codes(B, y)
        labels_matrix = 2 * B_prim - 1
        for i in xrange(num_of_features):
            current_labels = labels_matrix[:, i]
            svm_classifier.fit(X, current_labels)
            hyperplanes_T[i] = svm_classifier.coef_
        if (B_prim == B).all():
            return hyperplanes_T
        B = binarize(np.dot(hyperplanes_T, X.T))
        B = B.T


def discriminative_binary_codes_adaptive_classification(X_train, y_train, X_test, num_of_features):
    classifier = get_svm()
    classifier.fit(X_train, y_train)
    y_test = classifier.predict(X_test)
    while True:
        hyperplanes_transposed = get_discr_binary_codes(X_test, y_test, num_of_features)
        new_X_train = np.sign(np.dot(hyperplanes_transposed, X_train.T))
        new_X_train = new_X_train.T
        classifier.fit(new_X_train, y_train)
        y_test_prim = classifier.predict(np.sign(np.dot(hyperplanes_transposed, X_test.T)).T)
        print sum(y_test_prim == y_test)
        if sum(y_test_prim == y_test) >= 0.95 * len(y_test):
            break
        else:
            y_test = y_test_prim
    return y_test
