# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:52:05 2017

@author: mcontrac
"""

import pandas

class GLMModel(object):
    """An object to build a GLM model from externally supplied coefficients"""

    __coefficients = None
    __features = None
    __rules = None
    intercept = None

    def __init__(self, coefficients_dataframe):
        """Initializes the features used in the model and their coefficients

        The input DataFrame should contain the columns ``feature`` and
        ``coefficient``. The intercept should be named ``(Intercept)`` in the
        ``feature`` column and its value should be in the ``coefficient``. All
        other names in the ``feature`` column must be unique.

        Args:
            **coefficients_dataframe**: A pandas DataFrame object containing the
            columns ``feature`` and ``coefficient``. All values of ``feature``
            must be unique.

        Return:
            A new GLMModel object with features and coefficients set

        Raises:
            **AttributeError**: If ``coefficients_dataframe`` does not contain
            the required columns\n
            **ValueError**: If ``feature`` column contains duplicates
        """
        if 'feature' not in coefficients_dataframe.columns:
            raise AttributeError('Coefficients data does not contain column named "feature"')
        if 'coefficient' not in coefficients_dataframe.columns:
            raise AttributeError('Coefficients data does not contain column named "coefficient"')
        if len(coefficients_dataframe['feature'].unique()) != len(coefficients_dataframe['feature']):
            raise ValueError('Column "feature" is not unique on all values')
        self.intercept = coefficients_dataframe.loc[coefficients_dataframe['feature'] == '(Intercept)']['coefficient']
        self.__coefficients = coefficients_dataframe.loc[~(coefficients_dataframe['feature'] == '(Intercept)')]
        self.__features = set(self.__coefficients['feature'].values)

    def score_data(self, obs):
        """Scores a single observation and makes a predicition

        This method takes the Xi values of a single observation as a pandas
        DataFrame with the feature names in a column. Basically, the DataFrame
        should have two columns, ``feature`` and ``xi``, with one row for each
        feature. The method applies the relevant coefficients and calculates the
        contribution of each feature, and returns that along with the predicted
        score.

        Args:
            **obs**: A pandas DataFrame containing the ``xi`` values for scoring
            based on the model. The DataFrame should have the column ``feature``
            with same values as the ones used in the model, and a column ``xi``.

        Return:
            A tuple with the predicted value of the model as the 0th element and
            a pandas DataFrame with the non-zero contributions as the 1th element

        Raises:
            **ValueError**: If the ``feature`` column does not contain the same
            values as the features required by the model
        """
        if len(obs['feature'].unique()) != len(obs['feature']):
            raise ValueError('Column "feature" is not unique on all values in input data')
        if not set(obs['feature']).issuperset(self.__features):
            raise ValueError('Input data does not contain all features in model')
        scoring_data = self.__coefficients.merge(obs[['feature','xi']], how='left', on='feature')
        scoring_data['contribution'] = scoring_data['xi'] * scoring_data['coefficient']
        return (self.intercept + scoring_data['contribution'].sum(), scoring_data[scoring_data['contribution'] != 0])

    def load_rules(self, rules_dict):
        """Defines rules for calculating Xi based on raw input

        The rules are input as a lambda or a function and contain the
        transformation logic from the raw variables to the model inputs. The
        data will be input as a pandas Series, with the raw variable names
        as index.\n\n

        Example rule: ``rules_dict[x_10_25] = lambda raw_data: 1 if 10 <
        raw_data['x'] <= 25 else 0``

        Args:
            **rules_dict**: A dictionary containing the variable names as keys
            and a lambda or a function as a value.

        Raises:
            **ValueError**: If the rules do not cover all the features
            required for the model
        """
        if not set(rules_dict.keys()).issuperset(self.__features):
            raise ValueError('Rules for all features are not defined')
        self.__rules = rules_dict

    def prep_data_and_score(self, obs):
        """Transforms the raw data using the rules and scores the observation

        This method takes a single observation as input, creates the required
        features for the model and calculates the Xi values for each, and then
        scores the observation. This function can be used as an input to the
        ``apply`` method of a DataFrame of observations that need to be scored.

        Args:
            **obs**: Raw data that will be input to the rules for creating
            the model features

        Return:
            A tuple with the predicted value of the model as the 0th element and
            a pandas DataFrame with the non-zero contributions as the 1th element

        Raises:
            **ValueError**: If rules have not been defined
        """
        if self.__rules is None:
            raise ValueError('No rules defined')
        model_data = pandas.DataFrame(columns=['feature', 'xi'])
        i = 0
        for column in self.__features:
            model_data.loc[i] = [column, self.__rules[column](obs)]
            i += 1
        return self.score_data(model_data)
