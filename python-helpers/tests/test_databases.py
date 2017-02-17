# -*- coding: utf-8 -*-
"""
Unit tests for methods in helpers.databases module
"""
import unittest

from numpy import inf, NaN

from helpers.databases import get_dnb_scores
from helpers.databases import get_sic_major_group

class test_dnb_scores(unittest.TestCase):

    def test_vanilla(self):
        scores = get_dnb_scores(608201141)
        self.assertIn('credit_score_pct', scores)
        self.assertIn('financial_score_pct', scores)
        self.assertIsInstance(scores['credit_score_pct'], float)
        self.assertIsInstance(scores['financial_score_pct'], float)

    def test_bad_dnb(self):
        scores = get_dnb_scores(123.456)
        # Have to use assertIn since NaN == NaN gives False
        self.assertIn(scores['credit_score_pct'], [NaN])
        self.assertIn(scores['financial_score_pct'], [NaN])

    def test_null_dnb(self):
        with self.assertRaises(TypeError):
            get_dnb_scores(None)

    def test_string_dnb(self):
        with self.assertRaises(TypeError):
            get_dnb_scores('123')


class test_sic_major_class(unittest.TestCase):

    def test_vanilla(self):
        self.assertEquals(get_sic_major_group('0100'), 'Agriculture, Forestry & Fishing')

    def test_unknown_sic(self):
        self.assertIsNone(get_sic_major_group('0001'))

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            get_sic_major_group('100')

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            get_sic_major_group(2824)

if __name__ == '__main__':
    unittest.main()
