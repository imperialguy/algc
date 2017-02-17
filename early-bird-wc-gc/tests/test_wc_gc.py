# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:13:33 2016

@author: mcontrac
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('../early_bird'))
from wc_gc import read_rate_lookup


class test_read_rate_lookup(unittest.TestCase):

    def test_uslh_false(self):
        df = read_rate_lookup(False)
        self.assertIn('final_rate', df)

    def test_uslh_true(self):
        df = read_rate_lookup(True)
        self.assertIn('final_rate', df)

    # Make sure that default value is not set for the input
    def test_uslh_missing(self):
        self.assertRaises(TypeError, read_rate_lookup)

if __name__ == '__main__':
    unittest.main()