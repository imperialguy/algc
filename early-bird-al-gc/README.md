# algc

(yourvirtualenv) <path_to_early-bird-al-gc>
> nosetests
test_fail_calculate_algc_outputs_invalid_model_inputs (test_al_gc.TestALGC) ... ok
test_fail_calculate_bounds_invalid_lower_coefficient (test_al_gc.TestALGC) ... ok
test_fail_calculate_bounds_invalid_premium (test_al_gc.TestALGC) ... ok
test_fail_calculate_bounds_invalid_sigma_squared_constant (test_al_gc.TestALGC) ... ok
test_fail_calculate_bounds_invalid_sigma_squared_result (test_al_gc.TestALGC) ... ok
test_fail_calculate_bounds_invalid_upper_coefficient (test_al_gc.TestALGC) ... ok
test_fail_calculate_ilf_invalid_fixed_occurence_limit (test_al_gc.TestALGC) ... ok
test_fail_calculate_ilf_invalid_ilf_input (test_al_gc.TestALGC) ... ok
test_fail_calculate_ilf_invalid_policy_limit (test_al_gc.TestALGC) ... ok
test_fail_calculate_ilf_invalid_state_group (test_al_gc.TestALGC) ... ok
test_fail_calculate_premium_invalid_balance_rate (test_al_gc.TestALGC) ... ok
test_fail_calculate_premium_invalid_balance_ratio (test_al_gc.TestALGC) ... ok
test_fail_calculate_projection_invalid_percentage_weight_class (test_al_gc.TestALGC) ... ok
test_fail_calculate_projection_invalid_projection_dict (test_al_gc.TestALGC) ... ok
test_fail_calculator_invalid_type_coefficient (test_al_gc.TestALGC) ... ok
test_fail_calculator_invalid_type_density (test_al_gc.TestALGC) ... ok
test_fail_calculator_invalid_type_eazi_dataframe (test_al_gc.TestALGC) ... ok
test_fail_calculator_unavailable_coefficient (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_custom_rule_features (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_density_constant (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_eazi (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_model_coefficients (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_model_covariances (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_model_inputs (test_al_gc.TestALGC) ... ok
test_fail_run_model_invalid_rules (test_al_gc.TestALGC) ... ok
test_pass_calculate_algc_outputs (test_al_gc.TestALGC) ... ok
test_pass_calculate_bounds (test_al_gc.TestALGC) ... ok
test_pass_calculate_projection (test_al_gc.TestALGC) ... ok
test_pass_calculator (test_al_gc.TestALGC) ... ok
test_pass_end_to_end_dynamic (test_al_gc.TestALGC) ... ok

Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
early_bird_al_gc.py                         0      0   100%
early_bird_al_gc\al_gc.py                 254     41    84%
early_bird_al_gc\fixture_generator.py     107      3    97%
-----------------------------------------------------------
TOTAL                                     361     44    88%
----------------------------------------------------------------------
Ran 30 tests in 9.635s

OK