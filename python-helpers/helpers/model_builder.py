# -*- coding: utf-8 -*-
"""
This file contains a generic GLM Model object which takes the coefficients and
the features as an input from a pandas DataFrame, and builds a model object to
score new observations.
"""

import inspect

import pandas

class GLMModel(object):
    """An object to build a GLM model from externally supplied coefficients"""

    __coefficients = None
    __features = None
    __rules = dict()
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
        self.intercept = coefficients_dataframe[coefficients_dataframe['feature'] == '(Intercept)'].squeeze()['coefficient']
        self.__coefficients = coefficients_dataframe.ix[~(coefficients_dataframe['feature'] == '(Intercept)')]
        self.__features = set(self.__coefficients['feature'].values)

    def score_data(self, obs):
        """Scores a single observation and makes a prediction

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

    def create_rule(self, feature, rule_function):
        """Create a rule to calculate feature from input data

        The feature is the target feature for which the rule is created and
        called during the `prep_data_and_score` function. The rule_function is
        the function that defines the rule for the feature.

        The first argument to the rule function will be the observation and the
        second argument will be the model data for that feature read from the
        coefficients DataFrame during the object creation.\n

        This will override any existing rule for the feature.

        Example:
            If, in the coefficients DataFrame, upper_limit and lower_limit are
            defined, a rule can be applied to multiple features with:\n
            .. code-block:: python

                model.create_rule('x_10_25', lambda data, model:
                                  1 if model['lower_limit'] < data['x']
                                  <= model['upper_limit'] else 0)

        Args:
            **feature**: The feature for which the rule is defined\n
            **rule_function**: A callable function with specification
            func(arg1, arg2)

        Raises:
            **ValueError**: If feature is not defined in the model, or arg
            specification of rule function is not correct
        """
        if not feature in self.__features:
            raise ValueError('Feature not defined in model')
        if len(inspect.getargspec(rule_function)[0]) != 2:
            raise ValueError('Arg specification for rule_function should be func(arg1, arg2)')
        self.__rules.update({feature: rule_function})


    def load_rules(self, rules_dict):
        """Defines rules for calculating Xi based on raw input

        The rules are input as a callable instance and contain the
        transformation logic from the raw variables to the model inputs. The
        functions should take two arguments, the first one is the input data and
        the second argument is the model data read from the coefficients
        DataFrame for that particular feature. Both will be input as a pandas
        Series, with the variable names as index.\n\n

        This will override any existing rule for the feature.

        Example:
            If, in the coefficients DataFrame, upper_limit and lower_limit are
            defined, a rule can be applied to multiple features with:\n
            .. code-block:: python

                rule = lambda data, model: 1 if model['lower_limit'] \\
                        < data['x'] <= model['upper_limit'] else 0
                rules_dict = {'x_0_10': rule, 'x_11_25': rule, 'x_25_inf': rule}
                model.load_rules(rules_dict)

        Args:
            **rules_dict**: A dictionary containing the variable names as keys
            and a callable with specification func(arg1, arg2) as a value.

        Raises:
            **ValueError**: If feature is not defined in the model, or arg
            specification of rule function is not correct
        """
        for key, value in rules_dict.items():
            if key not in self.__features:
                raise ValueError('Feature %s is not defined' % key)
            if (not callable(value)) or len(inspect.getargspec(value)[0]) != 2:
                raise ValueError('Arg specification for rule_function should be func(arg1, arg2)')
        self.__rules.update(rules_dict)

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
        if len(self.__rules) == 0:
            raise ValueError('No rules defined')
        model_data = pandas.DataFrame(columns=['feature', 'xi'])
        i = 0
        for column in self.__features:
            model_data.loc[i] = [column, self.__rules[column](obs,
                    self.__coefficients[
                            self.__coefficients['feature'] == column].squeeze()
                    )]
            i += 1
        return self.score_data(model_data)
