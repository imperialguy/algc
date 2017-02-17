# -*- coding: utf-8 -*-
"""
This file contains the model logic for the Early Bird GL GC model.
"""

import ConfigParser
import math
import re

import numpy
import pandas

from helpers.model_builder import GLMModel
from helpers.utils import setup_logging


# Inputs
user_duns_number = 60150745
user_zip_code = '08816'
user_division = 54
user_effective_date = pandas.datetime(2012, 1, 1)
user_sic_code = '1381'
user_predominant_state = 'TX'
user_exposure_type = 'Payroll'
user_exposure_size = 11111110000000
user_occurence_limit = 2250000
user_retention_amount = 0
user_claims_history = [10, 10, 10]


def transform_variable(variable):
    """Creates and returns a function to calculate the model feature from
    the input variable

    The function returned runs the calculation:\n
    ``log(min(max(variable, floor), cap) + 1)``

    Args:
        **variable**: The variable to use for the function definition

    Return:
        A callable of that executes the documented calculations
    """

    def feature_calculation(data, model):
        return math.log(min(max(data[variable], model['floor']), model['cap']) + 1)
    return feature_calculation

def run_model(model_inputs, model_coefficients_filename, rules_dict):
    """Runs the model based on the provided inputs

    Builds a GLMModel object from the external coefficients, loads the rules
    to convert apply the model coefficients based on the inputs and then runs
    the model based on the inputs provided.

    Args:
        **model_inputs**: A dictionary or DataFrame containing the variables
        required by the model as keys\n
        **model_coefficients_filename**: Path to file containing the model
        coefficients for the Worker's Comp GC model\n
        **rules_dict**: Dictionary with lambda functions to derive the features
        used by the model from the input variables

    Return:
        The predicted loss ratio for the account
    """
    gl_gc_model = GLMModel(pandas.read_csv(model_coefficients_filename))
    gl_gc_model.load_rules(rules_dict)
    gl_gc_model.create_rule('log_l_mean_clm_cnt_123', transform_variable('avg_claim_count'))
    gl_gc_model.create_rule('log_density', transform_variable('zip_density'))
    if model_inputs['exposure_type'] == 'Payroll':
        gl_gc_model.create_rule('payroll_ind_log_payroll_m', transform_variable('exposure_size'))
        gl_gc_model.create_rule('sales_ind_log_sales_m', lambda data, model: 0)
    else:
        gl_gc_model.create_rule('payroll_ind_log_payroll_m', lambda data, model: 0)
        gl_gc_model.create_rule('sales_ind_log_sales_m', transform_variable('exposure_size'))
    return gl_gc_model.prep_data_and_score(model_inputs)
    #return math.exp(gl_gc_model.prep_data_and_score(model_inputs)[0])


def get_sic_data(sic_code, mapping_table):
    """Looks up the data related to the SIC code from the mapping table

    The method returns a Series with the indices ``SIC_Class``, ``premops_1``,
    ``premops_2``, ``premops_3``, ``products_a``, ``products_b`` and
    ``products_c``. These values are for the model only an

    Args:
        **sic_code**: SIC code as a string of 4 digits\n
        **mapping_table**: File name of the mapping table

    Return:
        A pandas Series with the data for the SIC code, if found, or None

    Raises:
        **ValueError**: If sic_code is not a string with exactly 4 digits
    """
    if re.compile('^[0-9]{4}$').match(sic_code) is None:
        raise ValueError('SIC code should be a string with exactly 4 digits')
    mapping_data = pandas.read_csv(mapping_table, dtype={'SIC': str}).set_index('SIC')
    try:
        sic_data = mapping_data.loc[sic_code]
    except KeyError:
        sic_data = None
    return sic_data


def get_zip_density(zip_code, easi_data):
    """Returns the density of the ZIP code from the EASI data

    Args:
        **zip_code**: ZIP code for which density has to be looked up as a
        string of 5 digits\n
        **easi_data**: File containing the EASI data

    Return:
        The density of the ZIP code as a float, it it is found, or numpy.NaN
        in all other cases

    Raises:
        **ValueError**: If zip_code is not a string with exactly 5 digits
    """
    if re.compile('^[0-9]{5}$').match(zip_code) is None:
        raise ValueError('ZIP code should be a string with exactly 5 digits')
    easi = pandas.read_csv(easi_data, dtype={'zip_code': str}).set_index('zip_code')
    try:
        density = easi['density'][zip_code]
    except KeyError:
        density = numpy.NaN
    return density


def get_division_factors(division, division_factors_file):
    """Looks up the off balance factor and the rate need for the division

    Args:
        **division**: The division for which the factors have to be looked up\n
        **division_factors_file**: File name which contains the factors

    Return:
        A dicitionary with the keys ``off_balance_factor`` and ``rate_need``
        and their corresponding values
    """
    division_factors = pandas.read_csv(division_factors_file).set_index('division')
    return {'off_balance_factor': division_factors['off_balance_factor'][division],
            'rate_need': division_factors['rate_need'][division]
            }


def get_ilf_factors(retention_amount, occurence_limit, loss_cap, sic_data):
    columns = ['ilf_retention', 'ilf_limit', 'ilf_cap']
    index = ['premops_low', 'premops_medium', 'premops_high', 'products_low', 'products_medium', 'products_high']
    factors = pandas.DataFrame(columns=columns, index=index).join(sic_data.rename(index='weight'))


def main():
    config = ConfigParser.ConfigParser()
    config.read('config/model_config.config')
    app_log = setup_logging('gl_gc_logger', config.get('logger', 'log_file_name'))
    app_log.info('Scoring DUNS number: %d' % user_duns_number)


    sic_data = get_sic_data(user_sic_code, config.get('data_files', 'sic_data'))
    model_inputs = dict()
    model_inputs['division'] = user_division
    model_inputs['exposure_size'] = user_exposure_size / 1000000
    model_inputs['exposure_type'] = user_exposure_type
    model_inputs['predom_state'] = user_predominant_state
    model_inputs['sic_class'] = sic_data['SIC_Class']
    model_inputs['zero_loss_ind'] = 1 if sum(user_claims_history) == 0 else 0
    model_inputs['zip_density'] = get_zip_density(user_zip_code, config.get('data_files', 'easi_data'))
    model_inputs['avg_claim_count'] = sum(user_claims_history) / len(user_claims_history)
    predicted_loss, _ = run_model(model_inputs,
                                  config.get('data_files',
                                             'model_coefficients_file'),
                                  eval(config.get('model_rules', 'rules')))
    division_factors = get_division_factors(user_division, config.get('data_files', 'division_factors'))
    ilf_factors = get_ilf_factors(user_retention_amount, user_occurence_limit,
                                  config.getint('constants', 'ilf_loss_cap'),
                                  config.get('data_files', 'sic_data'))
    midpoint = (predicted_loss
                * config.get('constants', 'loss_development_factor')
                * (ilf_factors['occurence_limit'] - ilf_factors['retention_amount'])
                * config.getint('constants', 'aggregate_limit')
                * division_factors['off_balance_factor']
                * (1 + division_factors['rate_need'])
                / ilf_factors['loss_cap'] * (1 - config.get('constants', 'expense_rate'))
                )


