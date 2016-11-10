import os
import unittest

from concept_drift.score_calculator.score_calculation import get_score_from_file
from settings import DATA_DIR


class TestScoreCalculation(unittest.TestCase):

    def test_score_calculation(self):
        classification_file = os.path.join(DATA_DIR, 'tests', 'solution.csv')
        labels_file = os.path.join(DATA_DIR, 'testLabels.csv')

        result = get_score_from_file(classification_file, labels_file)

        self.assertAlmostEqual(result, 0.7221, places=4)
