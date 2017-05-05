# -*- coding: utf-8 -*-
"""
Unit tests for the helpers.utils module
"""

import os
import re
import unittest

from helpers.utils import setup_logging

class test_logging(unittest.TestCase):

    def setUp(self):
        os.chdir('/tmp')

    def tearDown(self):
        for f in os.listdir(os.getcwd()):
            if re.search('test-log\d+\.log', f):
                os.remove(os.path.join(os.getcwd(), f))

    def test_vanilla(self):
        log = setup_logging('log1', 'test-log1.log')
        self.assertTrue(os.path.isfile('test-log1.log'))
        log.info('Test')
        with open('test-log1.log') as fh:
            log_lines = fh.readlines()
        self.assertIn('Test', log_lines[0])
        self.assertNotIn('INFO', log_lines[0])

    def test_print_level(self):
        log = setup_logging('log2', 'test-log2.log', print_level=True)
        self.assertTrue(os.path.isfile('test-log2.log'))
        log.info('Test')
        with open('test-log2.log') as fh:
            log_lines = fh.readlines()
        self.assertIn('Test', log_lines[0])
        self.assertIn('INFO', log_lines[0])


# register_model is tested along with the tests for the server
# It is only provided in the helpers library for easy use

if __name__ == '__main__':
    unittest.main(exit=False)
