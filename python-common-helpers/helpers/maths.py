# -*- coding: utf-8 -*-
"""
This file contains helper functions for mathemcatical and statistical methods.
It should not contain application specific helpers, utility helpers, database
querying or any other helper functions.
"""
import numpy

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
    rounded = round(number, max(0, ndigits))
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
    rounded = round(number, max(0, ndigits))
    return rounded if rounded <= number else rounded - (1 / (10 ** ndigits))

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
