from helpers.utils import (
    mproperty,
    setup_logging,
    ALGCJSONEncoder
)
from operator import getitem
from pyodbc import connect
from random import randint
import datetime
import logging
import xlwings
import random
import pandas
import numpy
import yaml
import json
import uuid
import os

logger = setup_logging(__file__, None, logging.DEBUG)


class ALGCDataGenerator(object):

    """ Class to generate the data for ALGC model(s)

    """

    def __init__(self, config):
        self.config = config

    @property
    def account_name(self):
        return self.config.default_account_name

    @property
    def effective_year(self):
        return datetime.datetime.strptime(
            self.effective_date,
            self.config.input_date_format).year

    @property
    def division(self):
        return self._random_list_picker(
            self.config.valid_divisions)

    @property
    def underwriter(self):
        return self.config.default_underwriter

    @property
    def sic_description(self):
        return self.sic_dataframe.loc[
            self.sic_dataframe.sic == self.sic].sic_industry.values[0]

    @property
    def state(self):
        return self.lists_dataframe.iloc[random.sample(
            list(self.lists_dataframe.index),
            1)[0]].state

    @property
    def zipcode(self):
        return int(self.eazi_dataframe.iloc[random.sample(
            list(self.eazi_dataframe.index),
            1)[0]].zip_code)

    @property
    def vehicle_count(self):
        return random.randint(1, self.config.vehicle_count_limit)

    @property
    def predominant_radius(self):
        return self._random_list_picker(
            self.config.valid_predominant_radiuses)

    @property
    def policy_limit(self):
        return self._random_list_picker(
            self.config.valid_policy_limits)

    @property
    def percentage_vehicles_extra_heavy(self):
        return 1 - sum(
            (self.percentage_vehicles_light,
             self.percentage_vehicles_medium,
             self.percentage_vehicles_heavy))

    @property
    def average_claim_count_in_last_3_years(self):
        return float(round(numpy.mean((
            self.number_of_claims_in_the_last_year,
            self.number_of_claims_in_the_2nd_last_year,
            self.number_of_claims_in_the_3rd_last_year)),
            self.config.average_claim_round_off_digits
        ))

    @property
    def financial_score(self):
        db_dnb, db_fpct = self.dnb_fpct

        return db_fpct

    @property
    def dnb(self):
        db_dnb, db_fpct = self.dnb_fpct

        return db_dnb

    @mproperty
    def dnb_fpct(self):
        with connect(self.netezza_dsn) as connection:
            while True:
                cursor = connection.execute(self.random_duns_sql)
                db_dnb, db_fpct = next(cursor)
                db_dnb, db_fpct = str(db_dnb), int(
                    db_fpct) if db_fpct is not None else db_fpct

                self.interface_sheet.range(
                    self.config.dnb_loc).value = db_dnb
                excel_financial_score = int(self.interface_sheet.range(
                    self.config.financial_score_loc).value)

                if excel_financial_score == db_fpct:
                    break

        return db_dnb, db_fpct

    @mproperty
    def sic(self):
        return int(self.sic_dataframe.iloc[random.sample(
            list(self.sic_dataframe.index), 1)[0]
        ].sic)

    @mproperty
    def number_of_claims_in_the_last_year(self):
        return random.randint(1, self.config.num_claims_limit)

    @mproperty
    def number_of_claims_in_the_2nd_last_year(self):
        return random.randint(0,
                              self.config.num_claims_limit -
                              self.number_of_claims_in_the_last_year)

    @mproperty
    def number_of_claims_in_the_3rd_last_year(self):
        return random.randint(0,
                              self.config.num_claims_limit - sum((
                                  self.number_of_claims_in_the_last_year,
                                  self.number_of_claims_in_the_2nd_last_year
                              )))

    @mproperty
    def percentage_vehicles_light(self):
        return random.randint(1, 100) * 0.01

    @mproperty
    def percentage_vehicles_medium(self):
        return random.randint(0,
                              100 - int(100 * self.percentage_vehicles_light)) * 0.01

    @mproperty
    def percentage_vehicles_heavy(self):
        return random.randint(0, 100 - int(sum(
            (100 * self.percentage_vehicles_light,
             100 * self.percentage_vehicles_medium)))) * 0.01

    @mproperty
    def effective_date(self):
        random_year = randint(
            self.config.random_year_start, self.config.random_year_end)
        random_month = randint(
            self.config.random_month_start, self.config.random_month_end)
        random_day = randint(
            self.config.random_day_start, self.config.random_day_end)
        return datetime.date(random_year, random_month, random_day
                             ).strftime(self.config.input_date_format)

    def _get_num_claims(self):
        return random.randint(1, self.config.num_claims_limit)

    def _get_random_large_number(self, num_digits):
        return int(str(uuid.uuid4().int)[:num_digits])

    def _random_list_picker(self, input_list):
        return input_list[random.randint(0, len(input_list) - 1)]

    def _setup_class(self):
        self.workbook_file = os.path.join(
            self.config.data_directory_path,
            self.config.excel_test_workbook_filename
        )
        self.workbook = xlwings.Book(self.workbook_file)
        self.interface_sheet = getitem(self.workbook.sheets,
                                       self.config.interface_sheet)


    def _load_constants(self):
        self.sic_dataframe = pandas.read_csv(self.config.sic_file_path)
        self.lists_dataframe = pandas.read_csv(self.config.lists_file_path)
        self.eazi_dataframe = pandas.read_csv(self.config.eazi_file_path)
        self.netezza_dsn = 'DSN={0}'.format(self.config.db_dsn)
        self.random_duns_sql = self.config.random_duns_sql.format(
            minimum_financial_score=self.config.minimum_financial_score,
            duns_limit=1)

    def to_dict(self):
        return dict(
            account_name=self.account_name,
            effective_date=self.effective_date,
            effective_year=self.effective_year,
            division=self.division,
            underwriter=self.underwriter,
            sic=self.sic,
            sic_description=self.sic_description,
            state=self.state,
            zipcode=self.zipcode,
            vehicle_count=self.vehicle_count,
            predominant_radius=self.predominant_radius,
            percentage_vehicles_light=self.percentage_vehicles_light,
            percentage_vehicles_medium=self.percentage_vehicles_medium,
            percentage_vehicles_heavy=self.percentage_vehicles_heavy,
            percentage_vehicles_extra_heavy=self.percentage_vehicles_extra_heavy,
            policy_limit=self.policy_limit,
            dnb=self.dnb,
            financial_score=self.financial_score,
            number_of_claims_in_the_last_year=self.number_of_claims_in_the_last_year,
            number_of_claims_in_the_2nd_last_year=self.number_of_claims_in_the_2nd_last_year,
            number_of_claims_in_the_3rd_last_year=self.number_of_claims_in_the_3rd_last_year,
            average_claim_count_in_last_3_years=self.average_claim_count_in_last_3_years
        )

    def dump(self):
        self._setup_class()
        self._load_constants()
        algc_input_data = self.to_dict()

        fixture_filename = '.'.join((datetime.datetime.now().strftime(
            self.config.fixture_file_name_timestamp_formatter),
            self.config.in_filename_appender,
            self.config.fixture_file_extension
        ))
        fixture_filepath = os.path.join(
            self.config.fixtures_directory_path,
            fixture_filename
        )

        with open(fixture_filepath, 'w') as f:
            if self.config.json_pretty_print:
                json.dump(algc_input_data, f,
                          indent=self.config.json_pretty_print_indent,
                          sort_keys=self.config.json_pretty_sort_keys_flag
                          )
            else:
                json.dump(algc_input_data, f)

        return fixture_filepath
