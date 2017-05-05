# -*- coding: utf-8 -*-
"""
Unit tests for the helpers.maths module
"""

import unittest

from numpy import inf, NaN
from pandas import DataFrame

from helpers.maths import get_discount_amount

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


if __name__ == '__main__':
    unittest.main()
