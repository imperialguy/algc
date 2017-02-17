# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:12:31 2016

@author: mcontrac
"""

import ConfigParser
import math

import numpy
import pandas

from helpers import setup_logging
from helpers import get_discount_amount
from helpers import get_dnb_scores
from helpers import get_sic_major_group
from helpers import round_down
from model_builder import GLMModel

# User inputs
user_duns_number = 608201141
user_division = 21
user_is_uslh = False
user_sic_code = '0111'
user_effective_date = pandas.datetime(2016, 9, 1)
user_total_projected_payroll = 10000000
user_estimated_clerical_payroll_ratio = 0.3
user_estimated_clerical_payroll = user_estimated_clerical_payroll_ratio * user_total_projected_payroll
user_estimated_non_clerical_payroll = user_total_projected_payroll - user_estimated_clerical_payroll
user_experience_mod = 0.97
input_data = pandas.DataFrame({'state': ['AK', 'CT', 'MD', 'KY', 'CA', 'CA', 'DE', 'AK'],
                               'class_code': ['6504', '4720', '2039', '6504', '8810', '6504', '0953', '9139'],
                               'payroll': [4000000, 500000, 1000000, 100000, 1000000, 200000, 200000, 0]})
input_history = pandas.DataFrame({'years_before': [1, 2, 3], 'ind_claim_count': [2, 2, 2], 'med_claim_count': [26, 19, 14]})

def read_rate_lookup(filename, is_uslh):
    """Reads the data from the rate_lookup.csv file into a pandas DataFrame

    The rate_lookup.csv file should contain the columns called ``state``,
    ``class_code``, ``final_rate``, ``final_rate_uslh`` and ``clerical_ind``.
    If the input division is 58-USLH, the ``final_rate`` column is dropped and
    the ``final_rate_uslh`` column is renamed to final_rate.
    Otherwise the ``final_rate_uslh`` column is dropped.

    Args:
        **is_uslh**: Boolean indicator whether the division is 58-USLH or not

    Return:
       A pandas DataFrame object with the state, class_code and final_rate
       columns
    """
    rate_lookup = pandas.read_csv(filename, index_col='lookup_key')
    if is_uslh:
        rate_lookup.drop('final_rate', axis=1, inplace=True)
        rate_lookup.rename(columns={'final_rate_uslh': 'final_rate'}, inplace=True)
    else:
        rate_lookup.drop('final_rate_uslh', axis=1, inplace=True)
    return rate_lookup


def read_discount_lookup(filename):
    """Reads the discount lookup data for the specifiec NCCI table number

    Args:
        **filename**: csv file from which to read the NCCI data

    Return:
        A pandas DataFrame containing the bucket as the index and the discount
        rates for each bucket
    """
    return pandas.read_csv(filename)


def read_state_rate_need_lookup(filename, division, effective_date, is_uslh):
    """Reads the fixed and variable rate need data for the input division and
    effective date

    The is_uslh indicator is only applicable to division 58. For all other
    divisions, the indicator is assumed to be False regardless of input.

    Args:
        **filename**: csv file containing the state rate need data\n
        **division**: The user input division\n
        **effective_date**: The user input effective date\n
        **is_uslh**: Boolean indicator for which division 58 rates to lookup

    Return:
        A pandas DataFrame with columns state, variable_rate_need,
        fix_rate_need and indicated_loss_ratio
    """
    state_rate_need = pandas.read_csv(filename, parse_dates=['effective_date', 'expiration_date'], infer_datetime_format=True)

    def keep_row(index):
        return (state_rate_need['division'][index] == division
                and state_rate_need['effective_date'][index] <= effective_date <= state_rate_need['expiration_date'][index]
                and state_rate_need['uslh_ind'][index] == is_uslh)

    return state_rate_need.select(keep_row).drop(['division', 'uslh_ind', 'effective_date', 'expiration_date'], axis=1)


def read_wcng_loss_ratio_lookup(filename, division, is_uslh):
    """Reads the WCNG average loss ratio for the division by state

    The is_uslh indicator is only applicable to division 58. For all other
    divisions, the indicator is assumed to be False regardless of input.

    Args:
        **filename**: csv file containing the WCNG loss ratio data\n
        **division**: The user input division\n
        **is_uslh**: Boolean indicator for which division 58 rates to lookup

    Return:
        A pandas DataFrame with columns state and avg_wcng_loss_ratio
    """
    wcng_loss_ratio = pandas.read_csv(filename)

    def keep_row(index):
        return (wcng_loss_ratio['division'][index] == division) and (wcng_loss_ratio['uslh_ind'][index] == is_uslh)

    return wcng_loss_ratio.select(keep_row).drop(['division', 'uslh_ind'], axis=1)


def read_cdf(filename, state):
    """Reads the CDFs for prior three years

    Args:
        **filename**: csv file containing the CDF data\n
        **state**: The state for which CDFs are to be read

    Return:
        A pandas DataFrame with columns ``prior_year`` and ``cdf``. Prior
        year refers to number of years prior to current year.
    """
    cdf_data = pandas.read_csv(filename)
    cdf_data['inverse_cdf'] = 1 / cdf_data['cdf']
    if state in cdf_data['state'].unique():
        return cdf_data[cdf_data['state'] == state].drop('state', axis=1)
    else:
        return cdf_data[cdf_data['state'].isnull()].drop('state', axis=1)


def get_monopolistic_states():
    """Returns a list of state codes that are monopolistic states"""
    return ['ND', 'OH', 'WA', 'WY']


def get_t9_states():
    """Returns a list of state codes that require T9 discount rates"""
    return ['AZ', 'FL', 'IA', 'ID', 'MA', 'NJ']


def merge_rate_lookup(input_data, rate_lookup_table):
    """Merges the ``clerical_ind`` and ``class_rate`` from the rate lookup to
    the input

    The function also calculates the class premium ,non-clerical and clerical
    payrolls for each input entry and also calculates the overall average
    clerical and non-clerical rates for the input provided. The function also
    adds the columns ``class_rate``, ``clerical_ind``, ``payroll_non_clerical``
    and ``payroll_clerical`` columns to the input data.

    Args:
        **input_data**: The  state, class code and payroll data input by the
        user as a DataFrame\n
        **rate_lookup_table**: The rates for calculating the class premium
        percents from payroll

    Return:
        A dictionary containing the average clerical rate (``avg_clerical_rate``) and
        the average non-clerical rate (``avg_non_clerical_rate``)
    """
    input_data['class_rate'] = input_data.apply(lambda row: rate_lookup_table['final_rate'][row['lookup_key']], axis=1)
    input_data['clerical_ind'] = input_data.apply(lambda row: rate_lookup_table['clerical_ind'][row['lookup_key']], axis=1)
    input_data['class_premium'] = input_data['payroll'] * input_data['class_rate']
    input_data['payroll_non_clerical'] = input_data['payroll'] * (1 - input_data['clerical_ind'])
    input_data['payroll_clerical'] = input_data['payroll'] * input_data['clerical_ind']
    avg_clerical_rate = sum(input_data['payroll_clerical'] * input_data['class_rate']) / input_data['payroll_clerical'].sum()
    avg_non_clerical_rate = sum(input_data['payroll_non_clerical'] * input_data['class_rate']) / input_data['payroll_non_clerical'].sum()
    return {'avg_clerical_rate': avg_clerical_rate, 'avg_non_clerical_rate': avg_non_clerical_rate}


def merge_wcng_lr_rate_need(payrolls, division, effective_date, is_uslh,
                            rate_need_file, wcng_lr_file):
    """Merges the payrolls data to the WCNG loss ratio and rate need data

    Note that this function returns a separate DataFrame object instead of
    merging inplace

    Args:
        **payrolls**: DataFrame containing the allocation ratio of each state\n
        **division**: The user input division\n
        **effective_date**: The user input effective date\n
        **is_uslh**: Boolean indicator for which division 58 rates to lookup\n
        **rate_need_file**: csv file containing the state rate need data\n
        **wcng_lr_file**: csv file containing the WCNG loss ratio data

    Return:
        A pandas DataFrame with all columns from ``payrolls`` along with
        ``avg_wcng_loss_ratio``, ``variable_rate_need``, ``fix_rate_need`` and
        ``indicated_loss_ratio`` columns

    """
    wcng_lr_data = read_wcng_loss_ratio_lookup(wcng_lr_file, division, is_uslh)
    rate_need_data = read_state_rate_need_lookup(rate_need_file, division, effective_date, is_uslh)
    return payrolls.merge(wcng_lr_data, how='left', on='state').merge(rate_need_data, how='left', on='state')


def calc_payroll_ratio(input_data):
    """Calculates the non-clerical and clerical payrolls for each state

    The function modifies the input dataframe and calculates the non-clerical
    payroll and clerical payroll columns for each row. It then calculates the
    total non-clerical and clerical payroll for each state and returns that as
    a DataFrame.

    Args:
        **input_data**: DataFrame containing the class premium, net, clerical
        and non-clerical payrolls for each state and class code

    Return:
        A pandas DataFrame with total class premium, net, non-clerical and
        clerical payrolls by state, and the ratio of non-clerical payroll for
        each state where the clerical payroll is missing
    """
    payrolls = input_data.groupby(by='state', as_index=False, sort=False).agg({'class_premium': 'sum',
                                                                               'payroll': 'sum',
                                                                               'payroll_non_clerical': 'sum',
                                                                               'payroll_clerical': 'sum'})
    payrolls['payroll_non_clerical_only'] = payrolls.apply(lambda row: row['payroll_non_clerical'] if row['payroll_clerical'] == 0 else 0,
                                                           axis=1)
    total_non_clerical = payrolls['payroll_non_clerical_only'].sum()
    payrolls['state_non_clerical_ratio'] = payrolls['payroll_non_clerical_only'] / total_non_clerical
    payrolls.drop('payroll_non_clerical_only', axis=1, inplace=True)
    return payrolls


def calc_allocate_clerical_payroll(payrolls, user_estimated_clerical_payroll):
    """Allocates the unentered clerical payroll to states based on non-clerical
    payroll ratio

    Uses the calculated non-clerical payroll ratio to allocate clerical payroll
    that was not entered by the user based on the user entered total estimated
    clerical payroll. The method modifies the payrolls DataFrame in place by
    adding the ``allocated_clerical_payroll`` column

    Args:
        **payrolls**: DataFrame containing the allocation ratio of each state\n
        **user_estimated_clerical_payroll**: User input total estimated
        clerical payroll
    """
    entered_clerical_payroll = payrolls['payroll_clerical'].sum()
    clerical_payroll_to_be_allocated = max(0, user_estimated_clerical_payroll - entered_clerical_payroll)
    payrolls['allocated_clerical_payroll'] = payrolls['state_non_clerical_ratio'] * clerical_payroll_to_be_allocated


def calc_clerical_class_premium(payrolls, rate_lookup_table):
    """Calculates the clerical class premium based on the allocated clerical
    payroll

    Determines the clerical rate to use from the rate table and calculates the
    class premium for clerical payroll based on the allocated clerical payroll.
    Modifies the payrolls DataFrame in place by adding the ``clerical_rate``
    and ``allocated_clerical_class_premium`` columns

    Args:
        **payrolls**: DataFrame containing the allocated clerical payroll for
        each state\n
        **rate_lookup_table**: Table containing the rate for each state and
        class code, with an boolean indicator for clerical vs non-clerical rate
    """
    clerical_rates = rate_lookup_table.loc[rate_lookup_table['clerical_ind'] == 1].set_index('state')
    payrolls['clerical_rate'] = payrolls['state'].map(clerical_rates['final_rate'])
    payrolls['allocated_clerical_class_premium'] = payrolls['clerical_rate'] * payrolls['allocated_clerical_payroll']


def calc_standard_premium(payrolls, user_experience_mod):
    """Calculates the standard premium for each state

    If a state is monopolistic, the experience mod is 1 else it is the user
    input experience mod. Monopolistic states are determined by the
    ``get_monopolistic_states()`` function. Modifies the payrolls DataFrame
    in place by adding the ``experience_mod``, ``standard_premium`` and
    ``standard_premium_ratio`` columns

    Args:
        **payrolls**: DataFrame containing the class premium by each state\n
        **user_experience_mod**: User input experience mod factor
    """
    monopolistic_states = get_monopolistic_states()
    payrolls['experience_mod'] = payrolls.apply(lambda row: user_experience_mod if row['state'] not in monopolistic_states else 1, axis=1)
    payrolls['standard_premium'] = payrolls['experience_mod'] * payrolls['class_premium']
    total_standard_premium = payrolls['standard_premium'].sum()
    payrolls['standard_premium_ratio'] = payrolls['standard_premium'] / total_standard_premium


def calc_missing_standard_premium(payrolls, avg_rates, user_experience_mod):
    """Returns the missing standard premiums to be allocated across the states

    Args:
        **payrolls**: DataFrame containing the clerical and non-clerical
        payroll by state\n
        **avg_rates**: Dictionary containing the average clerical and
        non-clerical rates for input\n
        **user_experience_mod**: User input experience mod factor

    Return:
        The total standard premium that is missing based on the inputs
    """
    missing_clerical_payroll = max(0, user_estimated_clerical_payroll - payrolls['payroll_clerical'].sum())
    missing_non_clerical_payroll = max(0, user_estimated_non_clerical_payroll - payrolls['payroll_non_clerical'].sum())
    allocated_clerical_class_premium = payrolls['allocated_clerical_class_premium'].sum()
    unknown_clerical_class_premium = (allocated_clerical_class_premium
                                      if allocated_clerical_class_premium > 0
                                      else avg_rates['avg_clerical_rate'] * missing_clerical_payroll)
    unknown_non_clerical_class_premium = missing_non_clerical_payroll * avg_rates['avg_non_clerical_rate']
    missing_clerical_standard_premium = unknown_clerical_class_premium * user_experience_mod
    missing_non_clerical_standard_premium = unknown_non_clerical_class_premium * user_experience_mod
    return missing_clerical_standard_premium + missing_non_clerical_standard_premium


def calc_allocated_standard_premium(payrolls, standard_premium_to_allocate):
    """Calcualtes the allocated the standard premiums for each state

    Distributes the missing standard premium to each state based on the
    standard premium ratio, and adds the calculated standard premium for the
    state to get the final allocated standard premium for the state. The
    function modifies the payrolls DataFrame in place by adding a
    ``allocated_standard_premium`` column

    Args:
        **payrolls**: DataFrame containing the standard premium value and ratio
        for each state\n
        **standard_premium_to_allocate**: The missing standard premium that
        needs to be distributed among the states
    """
    payrolls['allocated_standard_premium'] = (payrolls['standard_premium']
                                                + (payrolls['standard_premium_ratio'] * standard_premium_to_allocate))


def calc_premium_discount(payrolls, other_loadings, ncci_tier_files):
    """Calculates the premium discount to be applied to each state

    Reads the discount tables for NCCI state groups (currently only 7 and 9)
    and calculates the discount for each bucket within that group, totals it
    and puts it as ``premium_discount`` column in the ``payrolls`` DataFrame.
    The function also calculates the manual rate for each state as
    ``manual_rate`` column in the payrolls DataFrame.

    Args:
        **payrolls**: DataFrame containing the allocated standard premium for
        each state\n
        **other_loadings**: Other loadings factor for the rate calculations\n
        **ncci_tier_files**: A dict containing the NCCI tier number as key, and
        the filename as the value
    """
    ncci_table7 = read_discount_lookup(ncci_tier_files[7])
    ncci_table9 = read_discount_lookup(ncci_tier_files[9])
    t9_states = get_t9_states()

    def __discount_amount_helper(row):
        if row['state'] in t9_states:
            table = ncci_table9
        else:
            table = ncci_table7
        return get_discount_amount(row['allocated_standard_premium'], table)

    payrolls['premium_discount'] = payrolls.apply(__discount_amount_helper, axis=1)
    payrolls['manual_rate_pre_model'] = (1 + other_loadings) * (payrolls['allocated_standard_premium'] - payrolls['premium_discount'])
    payrolls['manual_rate'] = (1 + other_loadings) * (payrolls['standard_premium'] - payrolls['premium_discount'])


def calc_normalized_claim_counts(input_history, predom_state, aqi_data,
                                 total_class_premium, cdf_file):
    """Calculates the normalized indemnity and medical claim counts and ratio

    Uses the user input claim count history and the reference CDFs
    to calculate the normalized claim counts for the last 3 years,
    and calculates the indemnity to medical claim count ratio using
    the credibilty and global average from AQI profitability studies.

    Claim counts are calculated as 2 * claim count in prior year + claim counts
    in two years before that. CDF adjusted premium is also calculated similarly.
    Normalized claim counts are calculated by dividing the claim counts by the
    CDF adjusted premium in millions. The indemnity to medical claim ratio is
    calculated by adding the average respective claim frequency times the
    credibility (as obtained from AQI profitability study) to the claim counts,
    and then taking the ratio.

    Args:
        **input_history**: User input claim count history DataFrame\n
        **predom_state**: State whose CDFs are used\n
        **aqi_data**: A dictionary containing the keys ``credibility``,
        ``avg_indemenity_frequency_3yrs`` and ``avg_medical_frequency_3yrs``\n
        **total_class_premium**: Class premium value to use to calculate
        CDF adjusted premium\n
        **cdf_file**: csv file containing the CDF data

    Return:
        A pandas DataFrame containing the ``indemnity_claim_count``,
        ``medical_claim_count``,``cdf_adjusted_premium``,
        ``norm_indemnity_claim_count``, ``norm_medical_claim_count``
        and ``indemnity_medical_ratio`` as keys, with their corresponding values
    """
    __calc_claim_count = lambda column: input_history[column].sum() + input_history[input_history['years_before'] == 1][column]
    __norm_claim_count = lambda value, premium: value / (premium / 1000000)
    credibility = aqi_data['credibility']
    avg_indemnity_frequency_3yrs = aqi_data['avg_indemnity_frequency_3yrs']
    avg_medical_frequency_3yrs = aqi_data['avg_medical_frequency_3yrs']

    cdfs = read_cdf(cdf_file, predom_state)
    cdfs['cdf_premium'] = cdfs['inverse_cdf'] * total_class_premium
    cdf_premium_3yrs = cdfs['cdf_premium'].sum() + cdfs['prior_year' == 1]['cdf_premium'].sum()
    indemnity_claim_count = __calc_claim_count('ind_claim_count')
    medical_claim_count = __calc_claim_count('med_claim_count')
    norm_indemnity_claim_count = __norm_claim_count(indemnity_claim_count, cdf_premium_3yrs)
    norm_medical_claim_count = __norm_claim_count(medical_claim_count, cdf_premium_3yrs)
    indemnity_medical_ratio = ((indemnity_claim_count + (credibility * avg_indemnity_frequency_3yrs)) /
                                (medical_claim_count + (credibility * avg_medical_frequency_3yrs)))
    return pandas.DataFrame.from_dict(data={'indemnity_claim_count': indemnity_claim_count,
                                            'medical_claim_count': medical_claim_count,
                                            'cdf_adjusted_premium': cdf_premium_3yrs,
                                            'norm_indemnity_claim_count': norm_indemnity_claim_count,
                                            'norm_medical_claim_count': norm_medical_claim_count,
                                            'indemnity_medical_ratio': indemnity_medical_ratio
                                            }, orient='columns')


def calc_entered_payroll_ratios(input_data):
    """Calculates the entered clerical and non-clerical payroll ratios

    Entered clerical payroll ratio is defined as the clerical payroll entered
    divided by the total projected payroll. Max is 1.

    Entered non-clerical payroll ratio is defined as the non-clerical payroll
    entered divided the non-clerical payroll estimated. The estimated non-clerical
    payroll ratio is
    ``1 - max(entered_clerical_payroll_ratio, user_estimated_clerical_payroll_ratio)``
    If this is 0, the entered non-clerical payroll ratio is 0. Otherwise, max is
    1.

    Args:
        **input_data**: User input state, class code and payroll data after
        clerical and non-clerical payrolls have been calculated

    Return:
        A dictionary containing the entered ratios with keys as ``clerical`` and
        ``non_clerical``
    """
    entered_clerical_payroll_ratio = min(1, input_data['payroll_clerical'].sum() / user_total_projected_payroll)
    estimated_non_clerical_payroll_ratio = 1 - max(entered_clerical_payroll_ratio, user_estimated_clerical_payroll_ratio)
    if estimated_non_clerical_payroll_ratio > 0:
        estimated_total_non_clerical_payroll = estimated_non_clerical_payroll_ratio * user_total_projected_payroll
        entered_non_clerical_payroll_ratio = min(1, input_data['payroll_non_clerical'].sum() / estimated_total_non_clerical_payroll)
    else:
        entered_non_clerical_payroll_ratio = 0
    return {'clerical': entered_clerical_payroll_ratio,
            'non_clerical': entered_non_clerical_payroll_ratio}

def calc_diamond_bound_ratios(entered_clerical_payroll_ratio, entered_non_clerical_payroll_ratio,
                              bound_ratios_filename):
    """Calculates the upper and lower bound ratios for the diamond

    Args:
        **entered_clerical_payroll_ratio**: The ratio of clerical payroll to the
        total payroll entered\n
        **entered_non_clerical_payroll**: The ratio of non clerical payroll
        entered to the non clerical payroll estimated\n
        **bound_ratios_filename**: csv file containing the bound ratios for each
        division

    Return:
        A tuple whose 0th element is the lower bound ratio, and 1st element
        is the upper bound ratio. If ratios cannot be calculated, both are
        ``numpy.NaN``
    """
    if not 0.5 < entered_non_clerical_payroll_ratio < 1:
        base_ratio = entered_non_clerical_payroll_ratio
    elif (not 0.5 < entered_clerical_payroll_ratio < 1) and user_estimated_clerical_payroll_ratio == 1:
        base_ratio = entered_clerical_payroll_ratio
    else:
        return (numpy.NaN, numpy.NaN)

    bounds_base = (base_ratio - round_down(base_ratio, 1)) * 10
    bound_ratios = pandas.read_csv(bound_ratios_filename)
    bounds = bound_ratios.select(lambda ix: bound_ratios['ratio_lower_cap'][ix] < base_ratio <= bound_ratios['ratio_upper_cap'][ix])
    return ((bounds_base * bounds['lower_bound_delta']) + bounds['lower_bound'],
            (bounds_base * bounds['upper_bound_delta']) + bounds['upper_bound'])


def check_inputs(input_data, entered_ratios):
    """Checks whether inputs can be used by model for scoring

    Args:
        **input_data**: User input state, class code and payroll data after
        clerical and non-clerical payrolls have been calculated\n
        **entered_ratios**: The entered ratios dictionary returned by
        ``calc_entered_payroll_ratios(input_data)``

    Return:
        A tuple whose 0th element indicates whether inputs are usable or not,
        and if not, the 1st element provides the reason
    """
    if input_data['payroll'].sum() > (user_total_projected_payroll + 100):
        return (False, 'Input payroll exceeds total projected payroll')
    if input_data['payroll_clerical'].sum() > (user_total_projected_payroll * (user_estimated_clerical_payroll_ratio + 0.01)):
        return (False, 'Clerical payroll entry exceeds total clerical payroll estimate')

    estimated_non_clerical_payroll_ratio = 1 - max(entered_ratios['clerical'], user_estimated_clerical_payroll_ratio)
    if input_data['payroll_non_clerical'].sum() > (user_total_projected_payroll * (estimated_non_clerical_payroll_ratio + 0.01)):
        return (False, 'Non-clerical payroll entry exceeds total non-clerical payroll estimate')

    if ((user_estimated_clerical_payroll_ratio == 1 and entered_ratios['clerical'] > 0.6) or
        (user_estimated_clerical_payroll_ratio < 1 and entered_ratios['non_clerical'] > 0.6)):
        return (True, '')
    return (False, 'Not enough payroll data entered')

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
    wc_gc_model = GLMModel(pandas.read_csv(model_coefficients_filename))
    wc_gc_model.load_rules(rules_dict)
    return math.exp(wc_gc_model.prep_data_and_score(model_inputs)[0])


def main_wc_gc_model():
    config = ConfigParser.ConfigParser()
    config.read('config/model_config.config')
    app_log = setup_logging('wc_gc_logger', config.get('logger', 'log_file_name'))
    app_log.info('Scoring DUNS number: %d' % user_duns_number)

    rate_lookup_table = read_rate_lookup(config.get('data_files', 'rate_lookup'), user_is_uslh)
    input_data['lookup_key'] = input_data['state'] + input_data['class_code']
    avg_rates = merge_rate_lookup(input_data, rate_lookup_table)

    entered_ratios = calc_entered_payroll_ratios(input_data)
    inputs_valid, reason = check_inputs(input_data, entered_ratios)
    if not inputs_valid:
        return (numpy.NaN, numpy.NaN, numpy.NaN, reason)

    payrolls = calc_payroll_ratio(input_data)
    calc_allocate_clerical_payroll(payrolls, user_estimated_clerical_payroll)
    calc_clerical_class_premium(payrolls, rate_lookup_table)
    calc_standard_premium(payrolls, user_experience_mod)
    standard_premium_to_allocate = calc_missing_standard_premium(payrolls, avg_rates, user_experience_mod)
    calc_allocated_standard_premium(payrolls, standard_premium_to_allocate)
    calc_premium_discount(payrolls, config.getfloat('constants', 'other_loadings'),
                          eval(config.get('data_files', 'ncci_tier_files')))

    state_rate_data = merge_wcng_lr_rate_need(payrolls, user_division, user_effective_date, user_is_uslh,
                                              config.get('data_files', 'state_rate_need_lookup'),
                                              config.get('data_files', 'wcng_lr'))

    credit_scores = get_dnb_scores(user_duns_number,
                                   default_credit_score_pct=config.get('constants', 'default_duns_cs_pct'),
                                   default_financial_score_pct=config.get('constants', 'default_duns_fs_pct'))

    total_class_premium = input_data['class_premium'].sum()
    predom_state = input_data.groupby(by='state')['class_premium'].sum().idxmax(axis=1)
    model_inputs = calc_normalized_claim_counts(input_history, predom_state, eval(config.get('aqi', 'aqi_data')),
                                                total_class_premium, config.get('data_files', 'cdf_file'))

    model_inputs['credit_score_pct'] = credit_scores['credit_score_pct']
    model_inputs['financial_score_pct'] = credit_scores['financial_score_pct']
    model_inputs['payroll'] = user_total_projected_payroll
    model_inputs['major_group'] = get_sic_major_group(user_sic_code)

    predicted_lr, contributions = run_model(model_inputs, config.get('data_files', 'model_coefficients_file'),
                                            eval(config.get('model_rules', 'rules')))
    state_rate_data['target_pricing_deviation_factor'] = (((predicted_lr / state_rate_data['avg_wcng_loss_ratio'])
                                                          * state_rate_data['variable_rate_need'])
                                                          + state_rate_data['fix_rate_need'])
    state_rate_data['estimated_premium'] = state_rate_data['target_pricing_deviation_factor'] * state_rate_data['manual_rate_pre_model']
    output_midpoint = state_rate_data['estimated_premium'].sum()
    lower_ratio, upper_ratio = calc_diamond_bound_ratios(entered_ratios['clerical'], entered_ratios['non_clerical'],
                                                         config.get('data_files', 'bound_ratios'))
    return (output_midpoint * lower_ratio, output_midpoint, output_midpoint * upper_ratio, '')
