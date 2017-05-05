from early_bird_al_gc.fixture_generator import ALGCDataGenerator
from helpers.model_builder import GLMModel
from helpers.utils import (
    NestedDict,
    AttributeDict,
    setup_logging
)
try:
    import ConfigParser as configparser
except:
    import configparser
import logging
import pandas
import numpy
import yaml
import json
import math
import os


logger = setup_logging(__file__, None, logging.DEBUG)


def calculator(coefficient, eazi_dataframe, density_constant):
    """ Creates and returns a function to calculate the model feature from
    the input coefficient

    :param coefficient: coefficient that needs to be calculated
    :param eazi_dataframe: eazi csv loaded into pandas
    :param density_constant: density constant used for log density calculation
    :type coefficient: str
    :type eazi_dataframe: pandas.DataFrame
    :type density_constant: int
    :returns: a callable function to execute the coefficient's calculations
    :rtype: function

    """
    if not isinstance(coefficient, str):
        raise TypeError('coefficient should be a string')

    if not isinstance(eazi_dataframe, pandas.DataFrame):
        raise TypeError('eazi_dataframe should be a pandas DataFrame')

    if not isinstance(density_constant, int):
        raise TypeError('density_constant should be an integer')

    def calculate_log_medage(data, model):
        """ Calculates log of median age based on zipcode from model inputs

        :param data: model inputs
        :param model: parameters/frequency model coefficients loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: log of median age
        :rtype: float

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        return math.log(1 + eazi_dataframe.loc[
            eazi_dataframe.zip_code == data['zipcode']
        ].medage.values[0])

    def calculate_log_density(data, model):
        """ Calculates log of density based on zipcode from model inputs

        :param data: model inputs
        :param model: parameters/frequency model coefficients loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: log of density
        :rtype: float

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        return math.log(1 + min(density_constant,
                                eazi_dataframe.loc[
                                    eazi_dataframe.zip_code == data[
                                        'zipcode']].density.values[0]))

    def calculate_log_impute_exposure(data, model):
        """ Calculates log of impute exposure based on the number of vehicles

        :param data: model inputs
        :param model: parameters/frequency model coefficients loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: log of impute exposure
        :rtype: float

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        return math.log(data['vehicle_count'])

    def calculate_log_fpct(data, model):
        """ Calculates log of financial score percentage model input

        :param data: model inputs
        :param model: parameters/frequency model coefficients loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: log of financial score percentage
        :rtype: float

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        return math.log(1 + min(100, data['financial_score']))

    def calculate_log_legal_cnt_sum(data, model):
        """ Calculates log of legal count based on adverse filing count input

        :param data: model inputs
        :param model: parameters/frequency model coefficients loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: log of legal count
        :rtype: float

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        return math.log(1 + min(100, data['adverse_filing_count']))

    def calculate_count_deviation_0(data, model):
        """ Calculates count deviation 0 based on last 3 years claims

        :param data: model inputs
        :param model: parameters/frequency model coefficients loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: count deviation 0
        :rtype: float, int

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        return 1 if not numpy.average((
            data['number_of_claims_in_the_last_year'],
            data['number_of_claims_in_the_2nd_last_year'],
            data['number_of_claims_in_the_3rd_last_year'],
        )).astype(int) else 0

    def calculate_count_deviation_4(data, model):
        """ Calculates count deviation 4 based on last 3 years claims and fitted 
        result from frequency model calculations

        :param data: model inputs
        :param model: parameters model coefficients only loaded in pandas
        :type data: dict
        :type model: pandas.DataFrame
        :returns: count deviation 4
        :rtype: float, int

        """
        if not isinstance(data, dict):
            raise TypeError('data should be a dictionary')

        claims_average = numpy.average((
            data['number_of_claims_in_the_last_year'],
            data['number_of_claims_in_the_2nd_last_year'],
            data['number_of_claims_in_the_3rd_last_year'],
        ))

        return 0 if not claims_average else math.log(
            1 + claims_average / data['fitted_result'])

    calculator_dict = dict(cnt_deviation_0=calculate_count_deviation_0,
                           cnt_deviation_4=calculate_count_deviation_4,
                           log_impute_exposure=calculate_log_impute_exposure,
                           log_fpct=calculate_log_fpct,
                           log_legal_cnt_sum=calculate_log_legal_cnt_sum,
                           log_medage=calculate_log_medage,
                           log_density=calculate_log_density
                           )

    return calculator_dict[coefficient]


def get_config_and_rules(get_fixture_config=False):
    """ Gets the configuration and rules

    :param get_fixture_config: a flag to get fixture configuration only
    :type get_fixture_config: bool
    :returns: processor configuration and rules or fixture configuration
    :rtype: dict, tuple

    """
    if not isinstance(get_fixture_config, bool):
        raise TypeError('get_fixture_config should be a boolean')

    root_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..'))

    config_dir = os.path.join(root_dir, 'config')
    config_file = os.path.join(config_dir, 'algc.yml')

    with open(config_file, 'r') as f:
        main_config = AttributeDict(yaml.load(f))

    if get_fixture_config:
        fixture_config = AttributeDict(pandas.concat([
            pandas.Series(main_config.fixtures),
            pandas.Series(main_config.files),
            pandas.Series(main_config.database),
            pandas.Series(main_config.excel)]
        ).to_dict())

        fixture_config.data_directory_path = os.path.join(
            root_dir,
            fixture_config.data_directory
        )
        fixture_config.fixtures_directory_path = os.path.join(
            root_dir,
            fixture_config.fixtures_directory
        )
        fixture_config.logs_directory_path = os.path.join(
            root_dir,
            fixture_config.logs_directory
        )

        fixture_config.sic_file_path = os.path.join(
            fixture_config.data_directory_path,
            fixture_config.sic_file_name
        )
        fixture_config.eazi_file_path = os.path.join(
            fixture_config.data_directory_path,
            fixture_config.eazi_file_name
        )
        fixture_config.lists_file_path = os.path.join(
            fixture_config.data_directory_path,
            fixture_config.lists_file_name
        )

        return fixture_config

    processor_config = AttributeDict(pandas.concat([
        pandas.Series(main_config.processor),
        pandas.Series(main_config.files)]
    ).to_dict())

    rules_filepath = os.path.join(
        config_dir,
        processor_config.model_rules_configuration_filename)

    with open(rules_filepath, 'r') as f:
        rules_config = configparser.ConfigParser()
        try:
            # python 3.x
            rules_config.read_file(f)
        except AttributeError:
            # python 2.x
            rules_config.readfp(f)
        rules_dict = eval(rules_config.get('rules', 'rules'))

    processor_config.data_directory_path = os.path.join(
        root_dir,
        processor_config.data_directory
    )
    processor_config.logs_directory_path = os.path.join(
        root_dir,
        processor_config.logs_directory
    )

    eazi_csv = os.path.join(
        processor_config.data_directory_path,
        processor_config.eazi_file_name
    )
    sic_csv = os.path.join(
        processor_config.data_directory_path,
        processor_config.sic_file_name
    )
    state_group_csv = os.path.join(
        processor_config.data_directory_path,
        processor_config.state_group_file_name
    )
    ilf_csv = os.path.join(
        processor_config.data_directory_path,
        processor_config.ilf_file_name
    )
    balance_rate_csv = os.path.join(
        processor_config.data_directory_path,
        processor_config.balance_rate_file_name
    )
    balance_ratio_csv = os.path.join(
        processor_config.data_directory_path,
        processor_config.balance_ratio_file_name
    )

    processor_config.eazi_dataframe = pandas.read_csv(eazi_csv)
    processor_config.sic_dataframe = pandas.read_csv(sic_csv)
    processor_config.state_group_dataframe = pandas.read_csv(state_group_csv)
    processor_config.ilf_dataframe = pandas.read_csv(ilf_csv)
    processor_config.balance_rate_dataframe = pandas.read_csv(balance_rate_csv)
    processor_config.balance_ratio_dataframe = pandas.read_csv(
        balance_ratio_csv)

    return processor_config, rules_dict


def create_rule(
        glm_model,
        feature,
        eazi_dataframe,
        density_constant):
    """ Creates a new rule inside the GLM model

    :param glm_model: a generic Generalized Linear Model
    :param feature: a certain feature that needs to be calculated
    :param eazi_dataframe: eazi csv loaded into pandas library
    :param density_constant: density constant used to calculate log density
    :type glm_model: helpers.model_builder.GLMModel
    :type feature: str
    :type eazi_dataframe: pandas.DataFrame
    :type density_constant: int
    :returns: a generic Generalized Linear Model object
    :rtype: helpers.model_builder.GLMModel

    """
    if not isinstance(glm_model, GLMModel):
        raise TypeError('glm_model should be an instance of GLMModel')

    if not isinstance(feature, str):
        raise TypeError('feature should be a string')

    if not isinstance(eazi_dataframe, pandas.DataFrame):
        raise TypeError('eazi_dataframe should be a Pandas DataFrame')

    if not isinstance(density_constant, int):
        raise TypeError('density_constant should be an integer')

    glm_model.create_rule(feature, calculator(
        feature, eazi_dataframe, density_constant))

    return glm_model


def run_model(
        model_inputs,
        model_covariances_dataframe,
        model_coefficients_dataframe,
        eazi_dataframe,
        density_constant,
        rules_dict,
        custom_rule_features,
        calculate_sigma_squared=False):
    """ Runs the ALGC GLM model to return the fitted and sigma squared results

    :param model_inputs: model inputs
    :param model_covariances_dataframe: model covariances loaded into pandas
    :param model_coefficients_dataframe: model coefficients loaded into pandas
    :param eazi_dataframe: eazi csv loaded into pandas
    :param density_constant: density constant used to calculate log density
    :param rules_dict: rules configuration for parameters/frequency models
    :param custom_rule_features: rules that are overriden by custom functions
    :param calculate_sigma_squared: flag to enable sigma squared calculation
    :type model_inputs: dict
    :type model_covariances_dataframe: pandas.DataFrame
    :type model_coefficients_dataframe: pandas.DataFrame
    :type eazi_dataframe: pandas.DataFrame
    :type density_constant: int
    :type rules_dict: dict
    :type custom_rule_features: list
    :type calculate_sigma_squared: bool
    :returns: fitted and sigma squared results for frequency/parameters models
    :rtype: tuple

    """
    if not isinstance(model_inputs, dict):
        raise TypeError('model_inputs should be a dictionary')

    if not isinstance(model_covariances_dataframe, pandas.DataFrame):
        raise TypeError(
            'model_covariances_dataframe should be a Pandas DataFrame')

    if not isinstance(model_coefficients_dataframe, pandas.DataFrame):
        raise TypeError(
            'model_coefficients_dataframe should be a Pandas DataFrame')

    if not isinstance(eazi_dataframe, pandas.DataFrame):
        raise TypeError(
            'eazi_dataframe should be a Pandas DataFrame')

    if not isinstance(density_constant, int):
        raise TypeError('density_constant should be an integer')

    if not isinstance(rules_dict, dict):
        raise TypeError('rules_dict should be a dictionary')

    if not isinstance(custom_rule_features, list):
        raise TypeError('custom_rule_features should be a list')

    if not isinstance(calculate_sigma_squared, bool):
        raise TypeError('calculate_sigma_squared should be a boolean')

    algc_glm_model = GLMModel(model_coefficients_dataframe)
    algc_glm_model.load_rules(rules_dict)

    for feature in custom_rule_features:
        algc_glm_model = create_rule(algc_glm_model, feature,
                                     eazi_dataframe, density_constant)

    predicted_loss, scoring_data = algc_glm_model.prep_data_and_score(
        model_inputs)

    fitted_result = numpy.exp(scoring_data.coefficient.dot(
        scoring_data.xi))

    sigma_squared_result = scoring_data.xi.dot(
        model_covariances_dataframe).dot(scoring_data.xi
                                         ) if calculate_sigma_squared else None

    return (fitted_result, sigma_squared_result)


def calculate_ilf_helper(
        ilf_dataframe,
        state_group_dataframe,
        state,
        policy_limit,
        ilf_weight_class):
    """ Helper function for ILF related calculations

    :param ilf_dataframe: ILF csv loaded into pandas as a DataFrame
    :param state_group_dataframe: state_groups csv loaded into pandas
    :param state: state for which ILF needs to be calculated
    :param policy_limit: policy limit supplied in the model inputs
    :param ilf_weight_class: ILF weight class
    :type ilf_dataframe: pandas.DataFrame
    :type state_group_dataframe: pandas.DataFrame
    :type state: str
    :type policy_limit: int
    :type ilf_weight_class: str
    :returns: cumulative ILF value for a certain state and weight class
    :rtype: float

    """
    if not isinstance(ilf_dataframe, pandas.DataFrame):
        raise TypeError('ilf_dataframe should be a Pandas DataFrame')

    if not isinstance(state_group_dataframe, pandas.DataFrame):
        raise TypeError('state_group_dataframe should be a Pandas DataFrame')

    if not isinstance(state, str) or len(state) != 2:
        raise TypeError('Invalid state')

    if not isinstance(policy_limit, int):
        raise TypeError('policy_limit should be an integer')

    if not isinstance(ilf_weight_class, str):
        raise TypeError('ilf_weight_class should be a string')

    return sum(ilf_dataframe[(
        ilf_dataframe.state_group == int(
            state_group_dataframe[
                state_group_dataframe.state == state
            ].state_group)
    ) & (ilf_dataframe.limit == policy_limit
         )][ilf_weight_class])


def calculate_ilf(
        ilf_dataframe,
        state_group_dataframe,
        state,
        policy_limit,
        fixed_occurence_limit,
        ilf_weight_class):
    """ Calculates the ILF

    :param ilf_dataframe: ILF csv loaded into pandas as a DataFrame
    :param state_group_dataframe: state_groups csv loaded into pandas
    :param state: state for which ILF needs to be calculated
    :param policy_limit: policy limit supplied in the model inputs
    :param fixed_occurence_limit: policy limit constant
    :param ilf_weight_class: ILF weight class
    :type ilf_dataframe: pandas.DataFrame
    :type state_group_dataframe: pandas.DataFrame
    :type state: str
    :type policy_limit: int
    :type fixed_occurence_limit: int
    :type ilf_weight_class: str
    :returns: ILF ratio for a certain state and weight class
    :rtype: float

    """
    if not isinstance(ilf_dataframe, pandas.DataFrame):
        raise TypeError('ilf_dataframe should be a Pandas DataFrame')

    if not isinstance(state_group_dataframe, pandas.DataFrame):
        raise TypeError('state_group_dataframe should be a Pandas DataFrame')

    if not isinstance(state, str) or len(state) != 2:
        raise TypeError('Invalid state')

    if not isinstance(policy_limit, int):
        raise TypeError('policy_limit should be an integer')

    if not isinstance(fixed_occurence_limit, int):
        raise TypeError('fixed_occurence_limit should be an integer')

    if not isinstance(ilf_weight_class, str):
        raise TypeError('ilf_weight_class should be a string')

    return calculate_ilf_helper(
        ilf_dataframe,
        state_group_dataframe,
        state,
        policy_limit,
        ilf_weight_class
    ) / calculate_ilf_helper(
        ilf_dataframe,
        state_group_dataframe,
        state,
        fixed_occurence_limit,
        ilf_weight_class
    )


def calculate_bounds_helper(
        sigma_square_root_coefficient,
        sigma_squared_constant,
        sigma_squared_result,
        premium):
    """ Helper function used by `calculate_bounds` for calculating bounds

    :param sigma_square_root_coefficient: square root coefficient constant
    :param sigma_squared_constant: sigma squared constant
    :param sigma_squared_result: sigma squared calculation result
    :param premium: calculated premium
    :type sigma_square_root_coefficient: int
    :type sigma_squared_constant: float
    :type sigma_squared_result: float
    :type premium: float
    :returns: calculated bound value
    :rtype: float

    """
    if not isinstance(sigma_square_root_coefficient, int):
        raise TypeError('sigma_square_root_coefficient should be an integer')

    if not isinstance(sigma_squared_constant, float):
        raise TypeError(
            'sigma_squared_constant should be a floating point value')

    if not isinstance(sigma_squared_result, float):
        raise TypeError(
            'sigma_squared_result should be a floating point value')

    if not isinstance(premium, float):
        raise TypeError('premium should be a floating point value')

    return math.exp((sigma_square_root_coefficient * math.sqrt(
        sigma_squared_result)) - (
        sigma_squared_constant * sigma_squared_result)) * premium


def calculate_bounds(
        upper_sigma_square_root_coefficient,
        lower_sigma_square_root_coefficient,
        sigma_squared_constant,
        sigma_squared_result,
        premium):
    """ Calculates the upper and lower bounds

    :param upper_sigma_square_root_coefficient: square root coefficient constant
    :param lower_sigma_square_root_coefficient: square root coefficient constant
    :param sigma_squared_constant: sigma squared constant
    :param sigma_squared_result: sigma squared calculation result
    :param premium: calculated premium
    :type upper_sigma_square_root_coefficient: int
    :type lower_sigma_square_root_coefficient: int
    :type sigma_squared_constant: float
    :type sigma_squared_result: float
    :type premium: float
    :returns: upper and lower bounds in the same order
    :rtype: tuple

    """
    if not isinstance(upper_sigma_square_root_coefficient, int):
        raise TypeError(
            'upper_sigma_square_root_coefficient should be an integer')

    if not isinstance(lower_sigma_square_root_coefficient, int):
        raise TypeError(
            'lower_sigma_square_root_coefficient should be an integer')

    if not isinstance(sigma_squared_constant, float):
        raise TypeError(
            'sigma_squared_constant should be a floating point value')

    if not isinstance(sigma_squared_result, float):
        raise TypeError(
            'sigma_squared_result should be a floating point value')

    if not isinstance(premium, float):
        raise TypeError('premium should be a floating point value')

    return (calculate_bounds_helper(
        sigma_square_root_coefficient,
        sigma_squared_constant,
        sigma_squared_result,
        premium
    ) for sigma_square_root_coefficient in (
        upper_sigma_square_root_coefficient,
        lower_sigma_square_root_coefficient))


def calculate_premium(
        balance_ratio_dataframe,
        balance_rate_dataframe,
        parameters_fitted_result,
        ilf_result,
        division,
        expense_constant):
    """ Calculates premium

    :param balance_ratio_dataframe: balance ratio csv loaded into pandas
    :param balance_rate_dataframe: balance rate csv loaded into pandas
    :param parameters_fitted_result: fitted result for parameters model
    :param ilf_result: ILF calculation result
    :param division: division from model inputs
    :param expense_constant: expense constant from configuration file
    :type balance_ratio_dataframe: pandas.DataFrame
    :type balance_rate_dataframe: pandas.DataFrame
    :type parameters_fitted_result: float
    :type ilf_result: float
    :type division: int
    :type expense_constant: float
    :returns: calculated premium
    :rtype: float

    """
    if not isinstance(balance_ratio_dataframe, pandas.DataFrame):
        raise TypeError('balance_ratio_dataframe should be a Pandas DataFrame')

    if not isinstance(balance_rate_dataframe, pandas.DataFrame):
        raise TypeError('balance_rate_dataframe should be a Pandas DataFrame')

    if not isinstance(parameters_fitted_result, float):
        raise TypeError('parameters_fitted_result should be a floating value')

    if not isinstance(ilf_result, float):
        raise TypeError('ilf_result should be a floating value')

    if not isinstance(division, int):
        raise TypeError('division should be an integer')

    if not isinstance(expense_constant, float):
        raise TypeError('expense_constant should be a floating value')

    balance_ratio = float(balance_ratio_dataframe[
        balance_ratio_dataframe.division == division
    ].ratio)

    balance_rate = float(balance_rate_dataframe[
        balance_rate_dataframe.division == division
    ].rate)

    return parameters_fitted_result / (
        1 - expense_constant
    ) * (ilf_result) / balance_ratio * balance_rate


def calculate_projection(percentage_weight_class_dict, projection_dict):
    """ Calculate the projection

    :param percentage_weight_class_dict: weight class percentage dictionary
    :param projection_dict: projection dictionary
    :type percentage_weight_class_dict: dict
    :type projection_dict: dict
    :returns: calculated projection
    :rtype: float

    """
    if not isinstance(percentage_weight_class_dict, dict):
        raise TypeError('percentage_weight_class_dict should be a dictionary')

    if not isinstance(projection_dict, dict):
        raise TypeError('projection_dict should be a dictionary')

    return float(pandas.DataFrame(pandas.Series(
        percentage_weight_class_dict)).T.dot(pandas.DataFrame(
            pandas.Series(projection_dict))).values[0][0])


def calculate_algc_outputs(model_inputs):
    """ Calculates ALGC outputs

    :param model_inputs: ALGC interface inputs
    :type model_inputs: dict
    :returns: calculated early bird projections
    :rtype: dict

    """
    if not isinstance(model_inputs, dict):
        raise TypeError('model_inputs should be a dictionary')

    processor_config, rules_dict = get_config_and_rules(False)

    model_type_map = {
        processor_config.frequency_model_type:
        processor_config.frequency_custom_rule_features,
        processor_config.parameter_model_type:
        processor_config.parameters_custom_rule_features
    }

    coefficients_map = {
        processor_config.frequency_model_type:
        processor_config.frequency_coefficients_file,
        processor_config.parameter_model_type:
        processor_config.parameter_coefficients_file
    }

    model_inputs[
        'adverse_filing_count'] = processor_config.adverse_filing_count
    model_inputs['sic_class'] = processor_config.sic_dataframe.loc[
        processor_config.sic_dataframe.sic == model_inputs['sic']
    ].sic_class.values[0]

    model_covariances_filepath = os.path.join(
        processor_config.data_directory_path,
        processor_config.covariance_file_name)

    model_covariances_dataframe = pandas.read_csv(
        model_covariances_filepath,
        dtype={'feature': str}).set_index('feature')

    fitted_result_dict = NestedDict()
    sigma_squared_result_dict = dict()
    ilf_result_dict = dict()
    premium_result_dict = dict()
    upper_bound_dict = dict()
    lower_bound_dict = dict()
    percentage_weight_class_dict = dict()

    for weight_class in processor_config.valid_weight_classes:
        ilf_result_dict[weight_class] = calculate_ilf(
            processor_config.ilf_dataframe,
            processor_config.state_group_dataframe,
            model_inputs['state'],
            model_inputs['policy_limit'],
            processor_config.fixed_occurence_limit,
            processor_config.ilf_weight_class_map[weight_class]
        )

        for model_type in processor_config.model_types:
            model_inputs['class'] = weight_class
            calculate_sigma_squared = True if model_type == \
                processor_config.parameter_model_type else False

            model_coefficients_filepath = os.path.join(
                processor_config.data_directory_path,
                coefficients_map.get(model_type))
            model_coefficients_dataframe = pandas.read_csv(
                model_coefficients_filepath)

            fitted_result, sigma_squared_result = run_model(
                model_inputs,
                model_covariances_dataframe,
                model_coefficients_dataframe,
                processor_config.eazi_dataframe,
                processor_config.density_constant,
                rules_dict.get(model_type),
                model_type_map.get(model_type),
                calculate_sigma_squared
            )

            if model_type == processor_config.frequency_model_type:
                model_inputs['fitted_result'] = fitted_result
            elif model_type == processor_config.parameter_model_type:
                sigma_squared_result_dict[weight_class] = sigma_squared_result

                percentage_weight_class_dict[weight_class] = model_inputs[
                    processor_config.percentage_weight_class_map[
                        weight_class]]
                premium_result_dict[weight_class] = calculate_premium(
                    processor_config.balance_ratio_dataframe,
                    processor_config.balance_rate_dataframe,
                    fitted_result,
                    ilf_result_dict[weight_class],
                    model_inputs['division'],
                    processor_config.expense_constant
                )
                (upper_bound_dict[weight_class],
                 lower_bound_dict[weight_class]) = calculate_bounds(
                    processor_config.upper_sigma_square_root_coefficient,
                    processor_config.lower_sigma_square_root_coefficient,
                    processor_config.sigma_squared_constant,
                    sigma_squared_result,
                    premium_result_dict[weight_class]
                )

            fitted_result_dict[model_type][
                weight_class] = fitted_result

    mid_point_projection = calculate_projection(
        percentage_weight_class_dict, premium_result_dict)
    per_vehicle_projection = mid_point_projection / \
        model_inputs['vehicle_count']

    projection_dict = dict(
        mid_point=mid_point_projection,
        per_vehicle=per_vehicle_projection,
        lower_bound=calculate_projection(
            percentage_weight_class_dict, lower_bound_dict),
        upper_bound=calculate_projection(
            percentage_weight_class_dict, upper_bound_dict)
    )

    return projection_dict


def main():
    fixture_config = get_config_and_rules(True)
    fixture__in_filepath = ALGCDataGenerator(
        fixture_config).dump() if fixture_config.generate else os.path.join(
        fixture_config.fixtures_directory_path,
        fixture_config.default_fixture_filename
    )
    fixture_filename, filextension = os.path.splitext(fixture__in_filepath)
    fixture_filename = fixture_filename.replace(
        fixture_config.in_filename_appender,
        fixture_config.out_filename_appender,
    )
    fixture__out_filepath = os.path.join(
        fixture_config.fixtures_directory_path,
        ''.join((fixture_filename, filextension))
    )

    logger.debug('model input fixture: {}'.format(fixture__in_filepath))

    with open(fixture__in_filepath, 'r') as f:
        model_inputs = json.load(f)

    algc_outputs = calculate_algc_outputs(model_inputs)

    with open(fixture__out_filepath, 'w') as f:
        json.dump(algc_outputs, f,
                  indent=fixture_config.json_pretty_print_indent,
                  sort_keys=fixture_config.json_pretty_sort_keys_flag
                  )

    logger.debug('model output fixture: {}'.format(fixture__out_filepath))

    if fixture_config.generate and fixture_config.remove_fixture:
        os.remove(fixture__in_filepath)


if __name__ == '__main__':
    main()
