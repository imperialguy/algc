# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:03:43 2016

@author: mcontrac
"""

import os
import sys
import unittest

from numpy import inf, NaN
from pandas import DataFrame

sys.path.insert(0, os.path.abspath('../early_bird'))
from helpers import get_discount_amount
from helpers import get_dnb_scores
from helpers import get_sic_major_group


class test_discount_amount(unittest.TestCase):

    def test_vanilla(self):
        buckets = DataFrame({'bucket': [0, 1, 2, 3],
                             'width': [10, 90, 900, inf],
                             'rate': [0.1, 0.2, 0.3, 0.4]})
        discount = int(get_discount_amount(250, buckets))
        self.assertEquals(discount, 64)

    def test_index_error(self):
        buckets = DataFrame({'bucket': [0, 1, 2, 3],
                             'width': [10, 90, 900, inf]})
        with self.assertRaises(KeyError):
            get_discount_amount(250, buckets)


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
        self.assertIN(scores['financial_score_pct'], [NaN])

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
