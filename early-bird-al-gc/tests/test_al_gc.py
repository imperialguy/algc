from early_bird_al_gc.fixture_generator import ALGCDataGenerator
from helpers.utils import (
    AttributeDict,
    setup_logging,
    time_it,
    json_load_byteified
)
from early_bird_al_gc.al_gc import (
    calculate_algc_outputs,
    get_config_and_rules,
    calculator,
    create_rule,
    run_model,
    calculate_ilf_helper,
    calculate_ilf,
    calculate_bounds_helper,
    calculate_bounds,
    calculate_premium,
    calculate_projection,
    calculate_algc_outputs
)
from operator import getitem
from pyodbc import connect
import unittest
import logging
import xlwings
import pytest
import pandas
import json
import sys
import os


class TestALGC(unittest.TestCase):

    def setUp(self):
        self.config = get_config_and_rules(True)

        log_file_path = os.path.join(
            self.config.logs_directory_path,
            self.config.tests_log_filename
        ) if self.config.log_to_file else None
        self.logger = setup_logging(
            __file__, log_file_path, logging.DEBUG)

        self.workbook = xlwings.Book(os.path.join(
            self.config.data_directory_path,
            self.config.excel_test_workbook_filename
        ))
        self.interface_sheet = getitem(self.workbook.sheets,
                                       self.config.interface_sheet)
        self.empty_list = list()
        self.empty_dict = dict()
        self.empty_dataframe = pandas.DataFrame(self.empty_dict)
        self.default_fixture_input = os.path.join(
            self.config.fixtures_directory_path,
            self.config.default_fixture_filename)

        with open(self.default_fixture_input) as f:
            self.default_model_inputs = json_load_byteified(f)

        fixture_filename, filextension = os.path.splitext(
            self.default_fixture_input)
        fixture_filename = fixture_filename.replace(
            self.config.in_filename_appender,
            self.config.out_filename_appender,
        )
        fixture__out_filepath = os.path.join(
            self.config.fixtures_directory_path,
            ''.join((fixture_filename, filextension))
        )

        with open(fixture__out_filepath) as f:
            self.projection_outputs = json.load(f)

    def test_fail_calculator_unavailable_coefficient(self):
        with self.assertRaises(KeyError):
            calculator(
                self.config.test_unavailable_coefficient,
                self.empty_dataframe,
                self.config.test_density_constant)

    def test_fail_calculator_invalid_type_coefficient(self):
        with self.assertRaises(TypeError):
            calculator(
                self.config.test_invalid_type_coefficient,
                self.empty_dataframe,
                self.config.test_density_constant)

    def test_fail_calculator_invalid_type_eazi_dataframe(self):
        with self.assertRaises(TypeError):
            calculator(
                self.config.test_available_coefficient,
                self.config.test_invalid_eazi_dataframe,
                self.config.test_density_constant)

    def test_fail_calculator_invalid_type_density(self):
        with self.assertRaises(TypeError):
            calculator(
                self.config.test_available_coefficient,
                self.empty_dataframe,
                self.config.test_invalid_density_constant)

    def test_pass_calculator(self):
        self.assertEqual(calculator(
            self.config.test_available_coefficient,
            self.empty_dataframe,
            self.config.test_density_constant).__name__,
            self.config.test_available_coefficient_function_name)

    def test_fail_run_model_invalid_model_inputs(self):
        with self.assertRaises(TypeError):
            run_model(None,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.config.test_density_constant,
                      self.empty_dict,
                      self.empty_list
                      )

    def test_fail_run_model_invalid_model_covariances(self):
        with self.assertRaises(TypeError):
            run_model(self.empty_dict,
                      None,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.config.test_density_constant,
                      self.empty_dict,
                      self.empty_list
                      )

    def test_fail_run_model_invalid_model_coefficients(self):
        with self.assertRaises(TypeError):
            run_model(self.empty_dict,
                      self.empty_dataframe,
                      None,
                      self.empty_dataframe,
                      self.config.test_density_constant,
                      self.empty_dict,
                      self.empty_list
                      )

    def test_fail_run_model_invalid_eazi(self):
        with self.assertRaises(TypeError):
            run_model(self.empty_dict,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      None,
                      self.config.test_density_constant,
                      self.empty_dict,
                      self.empty_list
                      )

    def test_fail_run_model_invalid_density_constant(self):
        with self.assertRaises(TypeError):
            run_model(self.empty_dict,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.config.test_invalid_density_constant,
                      self.empty_dict,
                      self.empty_list
                      )

    def test_fail_run_model_invalid_rules(self):
        with self.assertRaises(TypeError):
            run_model(self.empty_dict,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.config.test_density_constant,
                      self.config.test_invalid_rules,
                      self.empty_list
                      )

    def test_fail_run_model_invalid_custom_rule_features(self):
        with self.assertRaises(TypeError):
            run_model(self.empty_dict,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.empty_dataframe,
                      self.config.test_density_constant,
                      self.empty_dict,
                      self.config.test_invalid_custom_rule_features
                      )

    def test_fail_calculate_algc_outputs_invalid_model_inputs(self):
        with self.assertRaises(TypeError):
            calculate_algc_outputs(None)

    def test_pass_calculate_algc_outputs(self):
        self.assertEqual(calculate_algc_outputs(
            self.default_model_inputs),
            self.projection_outputs)

    def test_fail_calculate_projection_invalid_percentage_weight_class(self):
        with self.assertRaises(TypeError):
            calculate_projection(None,
                                 self.empty_dict
                                 )

    def test_fail_calculate_projection_invalid_projection_dict(self):
        with self.assertRaises(TypeError):
            calculate_projection(self.empty_dict, None)

    def test_pass_calculate_projection(self):
        self.assertEqual(
            calculate_projection(
                self.config.test_percentage_data,
                self.config.test_projection_data),
            self.config.test_projection_result
        )

    def test_fail_calculate_premium_invalid_balance_ratio(self):
        with self.assertRaises(TypeError):
            calculate_premium(
                None,
                self.empty_dataframe,
                self.config.test_parameters_fitted_result,
                self.config.test_ilf_result,
                self.config.test_division,
                self.config.test_expense_constant
            )

    def test_fail_calculate_premium_invalid_balance_rate(self):
        with self.assertRaises(TypeError):
            calculate_premium(
                self.empty_dataframe,
                self.config.test_bad_data_string,
                self.config.test_parameters_fitted_result,
                self.config.test_ilf_result,
                self.config.test_division,
                self.config.test_expense_constant
            )

    def test_fail_calculate_bounds_invalid_upper_coefficient(self):
        with self.assertRaises(TypeError):
            calculate_bounds(
                self.config.test_bad_data_string,
                self.config.test_valid_lower_coefficient,
                self.config.test_valid_sigma_squared_constant,
                self.config.test_valid_sigma_squared_result,
                self.config.test_valid_premium
            )

    def test_fail_calculate_bounds_invalid_lower_coefficient(self):
        with self.assertRaises(TypeError):
            calculate_bounds(
                self.config.test_valid_upper_coefficient,
                self.config.test_bad_data_string,
                self.config.test_valid_sigma_squared_constant,
                self.config.test_valid_sigma_squared_result,
                self.config.test_valid_premium
            )

    def test_fail_calculate_bounds_invalid_sigma_squared_constant(self):
        with self.assertRaises(TypeError):
            calculate_bounds(
                self.config.test_valid_upper_coefficient,
                self.config.test_valid_lower_coefficient,
                self.config.test_bad_data_string,
                self.config.test_valid_sigma_squared_result,
                self.config.test_valid_premium
            )

    def test_fail_calculate_bounds_invalid_sigma_squared_result(self):
        with self.assertRaises(TypeError):
            calculate_bounds(
                self.config.test_valid_upper_coefficient,
                self.config.test_valid_lower_coefficient,
                self.config.test_valid_sigma_squared_constant,
                None,
                self.config.test_valid_premium
            )

    def test_fail_calculate_bounds_invalid_premium(self):
        with self.assertRaises(TypeError):
            calculate_bounds(
                self.config.test_valid_upper_coefficient,
                self.config.test_valid_lower_coefficient,
                self.config.test_valid_sigma_squared_constant,
                self.config.test_valid_sigma_squared_result,
                self.config.test_bad_data_string,
            )

    def test_pass_calculate_bounds(self):
        (test_upper_bound_result,
         test_lower_bound_result) = calculate_bounds(
            self.config.test_valid_upper_coefficient,
            self.config.test_valid_lower_coefficient,
            self.config.test_valid_sigma_squared_constant,
            self.config.test_valid_sigma_squared_result,
            self.config.test_valid_premium
        )
        self.assertEqual((test_upper_bound_result,
                          test_lower_bound_result), (
            self.config.test_upper_bound_result,
            self.config.test_lower_bound_result))

    def test_fail_calculate_ilf_invalid_ilf_input(self):
        with self.assertRaises(TypeError):
            calculate_ilf(
                self.config.test_bad_data_string,
                self.empty_dataframe,
                self.config.test_state,
                self.config.test_policy_limit,
                self.config.test_fixed_occurence_limit,
                self.config.test_ilf_weight_class
            )

    def test_fail_calculate_ilf_invalid_state_group(self):
        with self.assertRaises(TypeError):
            calculate_ilf(
                self.empty_dataframe,
                self.config.test_bad_data_string,
                self.config.test_state,
                self.config.test_policy_limit,
                self.config.test_fixed_occurence_limit,
                self.config.test_ilf_weight_class
            )

    def test_fail_calculate_ilf_invalid_policy_limit(self):
        with self.assertRaises(TypeError):
            calculate_ilf(
                self.empty_dataframe,
                self.empty_dataframe,
                self.config.test_state,
                self.config.test_bad_data_string,
                self.config.test_fixed_occurence_limit,
                self.config.test_ilf_weight_class
            )

    def test_fail_calculate_ilf_invalid_fixed_occurence_limit(self):
        with self.assertRaises(TypeError):
            calculate_ilf(
                self.empty_dataframe,
                self.empty_dataframe,
                self.config.test_state,
                self.config.test_policy_limit,
                self.config.test_bad_data_string,
                self.config.test_ilf_weight_class
            )

    def test_fail_calculate_ilf_invalid_fixed_occurence_limit(self):
        with self.assertRaises(TypeError):
            calculate_ilf(
                self.empty_dataframe,
                self.empty_dataframe,
                self.config.test_state,
                self.config.test_policy_limit,
                self.config.test_fixed_occurence_limit,
                None
            )

    def test_pass_end_to_end_dynamic(self):
        if sys.platform != self.config.dynamic_fixture_platform:
            self.logger.debug(self.config.platform_error)
            return

        counter = 0
        while counter < self.config.fixture_test_limit:
            fixture__in_filepath = ALGCDataGenerator(
                self.config).dump() if self.config.generate else os.path.join(
                self.config.fixtures_directory_path,
                self.config.default_fixture_filename
            )
            fixture_filename, filextension = os.path.splitext(
                fixture__in_filepath)
            fixture_filename = fixture_filename.replace(
                self.config.in_filename_appender,
                self.config.out_filename_appender,
            )
            fixture__out_filepath = os.path.join(
                self.config.fixtures_directory_path,
                ''.join((fixture_filename, filextension))
            )

            with open(fixture__in_filepath, 'r') as f:
                model_inputs = AttributeDict(json.load(f))

            # self.logger.debug(
            #     'model input file: {}'.format(fixture__in_filepath))

            self.interface_sheet.range(
                self.config.account_name_loc
            ).value = model_inputs.account_name

            self.interface_sheet.range(
                self.config.effective_date_loc
            ).value = model_inputs.effective_date

            self.interface_sheet.range(
                self.config.division_loc
            ).value = model_inputs.division

            self.interface_sheet.range(
                self.config.underwriter_loc
            ).value = model_inputs.underwriter

            self.interface_sheet.range(
                self.config.sic_loc
            ).value = model_inputs.sic

            self.interface_sheet.range(
                self.config.sic_loc
            ).value = model_inputs.sic

            self.interface_sheet.range(
                self.config.state_loc
            ).value = model_inputs.state

            self.interface_sheet.range(
                self.config.state_loc
            ).value = model_inputs.state

            self.interface_sheet.range(
                self.config.zip_code_loc
            ).value = model_inputs.zipcode

            self.interface_sheet.range(
                self.config.vehicle_count_loc
            ).value = model_inputs.vehicle_count

            self.interface_sheet.range(
                self.config.predominant_radius_loc
            ).value = model_inputs.predominant_radius

            self.interface_sheet.range(
                self.config.percentage_vehicles_light_loc
            ).value = model_inputs.percentage_vehicles_light

            self.interface_sheet.range(
                self.config.percentage_vehicles_medium_loc
            ).value = model_inputs.percentage_vehicles_medium

            self.interface_sheet.range(
                self.config.percentage_vehicles_heavy_loc
            ).value = model_inputs.percentage_vehicles_heavy

            self.interface_sheet.range(
                self.config.percentage_vehicles_extra_heavy_loc
            ).value = model_inputs.percentage_vehicles_extra_heavy

            self.interface_sheet.range(
                self.config.policy_limit_loc
            ).value = model_inputs.policy_limit

            self.interface_sheet.range(
                self.config.dnb_loc
            ).value = model_inputs.dnb

            self.interface_sheet.range(
                self.config.number_of_claims_in_the_last_year_loc
            ).value = model_inputs.number_of_claims_in_the_last_year

            self.interface_sheet.range(
                self.config.number_of_claims_in_the_2nd_last_year_loc
            ).value = model_inputs.number_of_claims_in_the_2nd_last_year

            self.interface_sheet.range(
                self.config.number_of_claims_in_the_3rd_last_year_loc
            ).value = model_inputs.number_of_claims_in_the_3rd_last_year

            self.interface_sheet.range(
                self.config.number_of_claims_in_the_3rd_last_year_loc
            ).value = model_inputs.number_of_claims_in_the_3rd_last_year

            algc_outputs = AttributeDict(calculate_algc_outputs(model_inputs))
            self.assertEqual(
                round(algc_outputs.lower_bound,
                      self.config.test_decimal_points),
                float(self.interface_sheet.range(
                    self.config.lower_bound_loc).value))
            self.assertEqual(
                round(algc_outputs.mid_point, self.config.test_decimal_points),
                float(self.interface_sheet.range(
                    self.config.mid_point_loc).value))
            self.assertEqual(
                round(algc_outputs.upper_bound,
                      self.config.test_decimal_points),
                float(self.interface_sheet.range(
                    self.config.upper_bound_loc).value))
            self.assertEqual(
                round(algc_outputs.per_vehicle,
                      self.config.test_decimal_points),
                float(self.interface_sheet.range(
                    self.config.per_vehicle_loc).value))

            with open(fixture__out_filepath, 'w') as f:
                json.dump(algc_outputs, f,
                          indent=self.config.json_pretty_print_indent,
                          sort_keys=self.config.json_pretty_sort_keys_flag
                          )

            counter += 1


if __name__ == '__main__':
    unittest.main()
