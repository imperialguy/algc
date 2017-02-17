# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:12:31 2016

@author: mcontrac
"""

from __future__ import division
import logging
import re

import numpy
import pypyodbc

def setup_logging(logger_name, log_file, level=logging.INFO,
                  print_level=False, mode='a'):
    """Sets up a logger with format ``timestamp: message``

    This method is a simple wrapper to create a log file using Python's standard
    logging library. If ``logger_name`` already exists, the same logger will be
    used, else a new logger is created.

    Args:
        **logger_name**: Name of the logger object\n
        **log_file**: The full path to the log file\n
        **level**: The minimum level of messages the logger will consider.
        Default is ``logging.INFO``. See Python logging module documentation
        for details\n
        **print_level**: Flag indicating whether to print log level in the
        output. Default is ``False``\n
        **mode**: The mode to open the log file in. Default is 'a'.

    Return:
        Newly created logger object with the name provided
    """
    if print_level:
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s: %(message)s')

    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(formatter)

    stream_handler = logging.streamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_discount_amount(amount, discount_table):
    """Calculates a discount amount based on buckets

    The widths and the discount rate for each bucket are to be provided
    as a pandas DataFrame in the ``discount_table``. The function does a
    tax-like discount calculation by checking which bucket the amount falls
    in and sums the maximum discount for all buckets below that and the rate
    times the remainder of the amount for the bucket. Maximum discount is
    width times the rate. ``discount_table`` must be sorted by bucket,
    otherwise results are undefined.

    Args:
        **amount**: The amount on which the discount is to be calculated\n
        **dicount_table**: A pandas DataFrame containing the ``bucket``,
        ``width`` and ``rate`` columns for each bucket, sorted by bucket
    Return:
        The discount applicable to the provided amount
    """
    def calc_row_discount(row, bucket):
        if row['bucket'] < bucket:
            discount = row['width'] * row['rate']
        elif row['bucket'] == bucket:
            discount = (amount - bucket_cutoffs.shift()[row.name]) * row['rate']
        else:
            discount = 0
        return discount

    bucket_cutoffs = discount_table['width'].cumsum()
    bucket = discount_table['bucket'][numpy.argmax(bucket_cutoffs > amount)]
    return discount_table.apply(calc_row_discount, axis=1, args=(bucket,)).sum()

def get_dnb_scores(duns_number, default_credit_score_pct=numpy.NaN, default_financial_score_pct=numpy.NaN):
    """Returns financial and credit score percentiles for the client

    Connects to the CDH database and retrieves the latest data
    for the client from D&B CSAD table

    Args:
        **duns_number**: The D&B number of the client as an int\n
        **default_credit_score_pct**: The default value to use for the
        credit score percentile\n
        **default_financial_score_pct**: The default value to use for the
        financial score percentile\n

    Return:
        A dictionary containing the ``credit_score_pct`` and
        ``financial_score_pct`` keys and their values. Values are
        numpy.NaN if D&B number is not found in database
    """
    scores = {'credit_score_pct': default_credit_score_pct,
              'financial_score_pct': default_financial_score_pct}
    with pypyodbc.connect('Driver={NetezzaSQL};Server=paccmrntza01;Port=5480;'
                          'Database=ADM_LND_P;uid=mcontrac;pwd=DeadMead0w'
                          ) as connection:
        with connection.cursor() as cursor:
            results = cursor.execute("""
                select
                    duns_number,
                    cpct,
                    fpct
                from adm_lnd_p.admin.lnd_t_dnb_csad
                where duns_number = %d
                order by to_date('01'||datepll, 'ddMONyy') desc
                limit 1
            """ % duns_number)
            dnb_data = results.fetchone()
        if dnb_data:
            scores['credit_score_pct'] = dnb_data[1]
            scores['financial_score_pct'] = dnb_data[2]
        return scores


def get_sic_major_group(sic_code):
    """Returns the major industry class for the given SIC code

    Connects to the CDH database and retrieves the major class for
    the SIC code from the LND_T_SA_IBM_SIC_CODE table.

    Args:
        **sic_code**: The SIC code input as a string of 4 digits

    Return:
        The major class of the SIC code as a string, if found,
        else returns None

    Raises:
        **ValueError**: If the ``sic_code`` is not a string of 4 digits
    """
    if not (type(sic_code) == str and re.match('^[0-9]{4}$', sic_code)):
        raise ValueError('SIC code should be a string of 4 digits')
    with pypyodbc.connect('Driver={NetezzaSQL};Server=paccmrntza01;Port=5480;'
                          'Database=ADM_LND_P;uid=mcontrac;pwd=DeadMead0w'
                          ) as connection:
        with connection.cursor() as cursor:
            results = cursor.execute("""
                select
                    major_class
                from adm_lnd_p.admnzdba.lnd_t_sa_ibm_sic_code
                where sic_code = '%s'
                order by update_ts desc
                limit 1
            """ % sic_code)
            major_class = results.fetchone()
            if major_class:
                return major_class[0]
            else:
                return None


def round_up(number, ndigits=0):
    """Rounds up a number to given precision in decimal digits

    Args:
        **number**: Number to round up\n
        **ndigits**: Number of decimal digits in output. Negative values are
        treated as 0

    Return:
        Smallest number with ``ndigits`` decimal places greater than or equal to
        the input
    """
    ndigits = max(0, ndigits)
    rounded = round(number, ndigits)
    return rounded if rounded >= number else rounded + (1 / (10 ** ndigits))


def round_down(number, ndigits=0):
    """Rounds down a number to given precision in decimal digits

    Args:
        **number**: Number to round up\n
        **ndigits**: Number of decimal digits in output. Negative values are
        treated as 0

    Return:
        Largest number with ``ndigits`` decimal places less than or equal to
        the input
    """
    ndigits = max(0, ndigits)
    rounded = round(number, ndigits)
    return rounded if rounded <= number else rounded - (1 / (10 ** ndigits))
