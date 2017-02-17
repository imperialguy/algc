# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:01:29 2017

@author: mcontrac
"""

import math
import unittest

import numpy
import pandas

from .model_builder import GLMModel

class test_glm_model(unittest.TestCase):

    rules_dict = None
    data = None
    coeff = None

    def setUp(self):
        self.rules_dict = {
                          'x_0_10': lambda data, model: 0,
                          'x_10_20': lambda data, model: 0,
                          'y_A': lambda data, model: 1 if data['y'] == 'A' else 0,
                          'y_B': lambda data, model: 1 if data['y'] == 'B' else 0,
                          'y_C': lambda data, model: 1 if data['y'] == 'C' else 0,
                          'log_z': lambda data, model: math.log(data['z']) if 0 < data['z'] <= 100 else math.log(100)
                           }

        self.obs = {'x': 8,
                     'y': 'B',
                     'z': 16.83
                     }

        model_dict = {'feature': ['(Intercept)', 'x_0_10', 'x_10_20', 'y_A', 'y_B', 'y_C', 'log_z'],
                      'coefficient': [9.034, 0.12, 0.342, -1.343, 3.56, 0.92, -0.45],
                      'lower': [numpy.NaN, 0, 10, numpy.NaN, numpy.NaN, numpy.NaN, 1],
                      'upper': [numpy.NaN, 10, 20, numpy.NaN, numpy.NaN, numpy.NaN, 100]
                      }
        self.model_data = pandas.DataFrame(model_dict)
        self.model = GLMModel(self.model_data)

    def test_init(self):
        self.assertIsNotNone(GLMModel(self.model_data))

    def test_init_missing_features(self):
        with self.assertRaises(AttributeError):
            GLMModel(self.model_data.drop('feature', axis=1))

    def test_init_missing_coefficient(self):
        with self.assertRaises(AttributeError):
            GLMModel(self.model_data.drop('coefficient', axis=1))

    def test_duplicate_feature(self):
        self.model_data.loc[len(self.model_data)] = {'feature': 'x_0_10',
                                                     'coefficient': 0.435,
                                                     'lower': numpy.NaN,
                                                     'upper': numpy.NaN
                                                     }
        with self.assertRaises(ValueError):
            GLMModel(self.model_data)

    def test_load_rules(self):
        self.model.load_rules(self.rules_dict)
        self.assertGreater(len(self.model._GLMModel__rules), 0)

    def test_load_rules_bad_key(self):
        self.rules_dict['log_x'] = None
        with self.assertRaises(ValueError):
            self.model.load_rules(self.rules_dict)

    def test_load_rules_bad_value(self):
        self.rules_dict['log_z'] = None
        with self.assertRaises(ValueError):
            self.model.load_rules(self.rules_dict)

    def test_load_rules_bad_value_2(self):
        self.rules_dict['log_z'] = lambda data: data['z'] + 1
        with self.assertRaises(ValueError):
            self.model.load_rules(self.rules_dict)

    def test_create_rule(self):
        rule = lambda data, model: math.log(data['x']) if model['lower'] < data['x'] <= model['upper'] else 0
        self.model.create_rule('x_0_10', rule)
        self.assertIs(self.model._GLMModel__rules['x_0_10'], rule)

    def test_create_rule_bad_feature(self):
        rule = lambda data, model: 0
        with self.assertRaises(ValueError):
            self.model.create_rule('a', rule)

    def test_create_rule_bad_rule(self):
        rule = lambda data: 0
        with self.assertRaises(ValueError):
            self.model.create_rule('x_0_10', rule)

    def test_prep_data_and_score(self):
        rule = lambda data, model: data['x'] if model['lower'] < data['x'] <= model['upper'] else 0
        self.model.create_rule('x_0_10', rule)
        self.model.create_rule('x_10_20', rule)
        score = self.model.prep_data_and_score(self.obs)
        self.assertAlmostEquals(score[0], 12.283576646308779)
        self.assertEquals(len(score[1]), 3)

    def test_prep_data_and_score_no_rules(self):
        self.model._GLMModel__rules = dict()
        with self.assertRaises(ValueError):
            self.model.prep_data_and_score(self.obs)

    def test_score_data(self):
        obs = pandas.DataFrame({'feature': ['x_0_10', 'x_10_20', 'y_A', 'y_B', 'y_C', 'log_z'],
                                'xi': [0, 13, 1, 0, 0, math.log(100)]
                                })
        score = self.model.score_data(obs)
        self.assertAlmostEquals(score[0], 10.064673416305359)
        self.assertEquals(len(score[1]), 3)

    def test_score_data_duplicate_features(self):
        obs = pandas.DataFrame({'feature': ['x_0_10', 'x_0_10', 'y_A', 'y_B', 'y_C', 'log_z'],
                                'xi': [0, 13, 1, 0, 0, math.log(100)]
                                })
        with self.assertRaises(ValueError):
            self.model.score_data(obs)

    def test_score_data_missing_features(self):
        obs = pandas.DataFrame({'feature': ['x_0_10', 'x_0_10', 'y_A', 'y_B', 'log_z'],
                                'xi': [0, 13, 1, 0, math.log(100)]
                                })
        with self.assertRaises(ValueError):
            self.model.score_data(obs)


if __name__ == '__main__':
    unittest.main(exit=False)