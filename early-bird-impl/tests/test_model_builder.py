# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:01:29 2017

@author: mcontrac
"""

import math
import os
import sys
import unittest

import pandas

sys.path.insert(0, os.path.abspath('./early_bird'))
from model_builder import GLMModel

class test_glm_model(unittest.TestCase):

    rules_dict = None
    data = None
    coeff = None

    def setUp(self):
        self.rules_dict = {
                          'indemnity_claims_0_to_8': lambda data: 1 if 0 <= data['norm_indemnity_claim_count'] <= 8 else 0,
                          'indemnity_claims_8_to_12': lambda data: 1 if 8 < data['norm_indemnity_claim_count'] <= 12 else 0,
                          'indemnity_claims_12_to_18': lambda data: 1 if 12 < data['norm_indemnity_claim_count'] <= 18 else 0,
                          'indemnity_claims_18_to_35': lambda data: 1 if 18 < data['norm_indemnity_claim_count'] <= 35 else 0,
                          'indemnity_claims_35_to_70': lambda data: 1 if 35 <= data['norm_indemnity_claim_count'] <= 70 else 0,
                          'indemnity_claims_70_to_inf': lambda data: 1 if 70 < data['norm_indemnity_claim_count'] else 0,
                          'indemnity_claims_na': lambda data: 1 if data['norm_indemnity_claim_count'] is None else 0,
                          'medical_claims_0_to_10': lambda data: 1 if 0 <= data['norm_medical_claim_count'] <= 10 else 0,
                          'medical_claims_10_to_20': lambda data: 1 if 10 < data['norm_medical_claim_count'] <= 20 else 0,
                          'medical_claims_20_to_52_5': lambda data: 1 if 20 < data['norm_medical_claim_count'] <= 52.5 else 0,
                          'medical_claims_52_5_to_100': lambda data: 1 if 52.5 < data['norm_medical_claim_count'] <= 100 else 0,
                          'medical_claims_100_to_180': lambda data: 1 if 100 < data['norm_medical_claim_count'] <= 180 else 0,
                          'medical_claims_180_to_inf': lambda data: 1 if 180 < data['norm_medical_claim_count'] else 0,
                          'medical_claims_na': lambda data: 1 if data['norm_medical_claim_count'] is None else 0,
                          'indemnity_medical_ratio': lambda data: min(0.6, max(0.4, data['indemnity_medical_ratio'])),
                          'duns_cpct': lambda data: min(40, max(0, data['credit_score_pct'])),
                          'duns_fpct_large': lambda data: min(80, max(0, data['financial_score_pct'])) if data['payroll'] >= 2000000 else 0,
                          'duns_fpct_small': lambda data: min(80, max(0, data['financial_score_pct'])) if data['payroll'] < 2000000 else 0,
                          'major_group_FI': lambda data: 1 if data['major_group'] == 'Financial Institutions' else 0,
                          'major_group_HC': lambda data: 1 if data['major_group'] == 'Healthcare and Life Science' else 0,
                          'major_group_MF': lambda data: 1 if data['major_group'] == 'Manufacturing' else 0,
                          'major_group_RT': lambda data: 1 if data['major_group'] == 'Retail Trade' else 0,
                          'major_group_TC': lambda data: 1 if data['major_group'] == 'Technology' else 0,
                          'major_group_OT': lambda data: 1 if data['major_group'] not in ('Financial Institutions', 'Healthcare and Life Science', 'Manufacturing', 'Retail Trade', 'Technology')  else 0,
                          'major_group_FI_payroll': lambda data: min(100, max(8, math.log(data['payroll']))) if data['major_group'] == 'Financial Institutions' else 0,
                          'major_group_HC_payroll': lambda data: min(100, max(8, math.log(data['payroll']))) if data['major_group'] == 'Healthcare and Life Science' else 0,
                          'major_group_MF_payroll': lambda data: min(100, max(8, math.log(data['payroll']))) if data['major_group'] == 'Manufacturing' else 0,
                          'major_group_RT_payroll': lambda data: min(100, max(8, math.log(data['payroll']))) if data['major_group'] == 'Retail Trade' else 0,
                          'major_group_TC_payroll': lambda data: min(100, max(8, math.log(data['payroll']))) if data['major_group'] == 'Technology' else 0,
                          'major_group_OT_payroll': lambda data: min(100, max(8, math.log(data['payroll']))) if data['major_group'] not in ('Financial Institutions', 'Healthcare and Life Science', 'Manufacturing', 'Retail Trade', 'Technology')  else 0,
                          'log_payroll_cap': lambda data: min(100, max(8, math.log(data['payroll']))),
                          'history_clean_and_large': lambda data: 1 if (data['indemnity_claim_count'] == 0
                                                                        and data['payroll'] >= 2000000
                                                                        and data['cdf_adjusted_premium'] > 0
                                                                        ) else 0
                           }

        self.data = {'indemnity_claim_count': 8,
                     'medical_claim_count': 85,
                     'cdf_adjusted_premium': 1.683534843,
                     'norm_indemnity_claim_count': 4.751906404,
                     'norm_medical_claim_count': 50.48900554,
                     'indemnity_medical_ratio': 0.102076904,
                     'credit_score_pct': 88,
                     'financial_score_pct': 69,
                     'payroll': 10000000,
                     'major_group': 'Retail Trade'
                     }

        self.coeff = pandas.read_csv('E:/GitHub/early-bird-impl/data/wc_gc_coefficients.csv')

    def test_wc_gc(self):
        wc_gc_model = GLMModel(self.coeff)
        wc_gc_model.load_rules(self.rules_dict)
        self.assertAlmostEquals(wc_gc_model.prep_data_and_score(self.data)[0], math.log(0.736876876))

    def runTest(self):
        self.test_wc_gc()

if __name__ == '__main__':
    unittest.main()
