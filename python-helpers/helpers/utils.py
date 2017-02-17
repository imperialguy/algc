# -*- coding: utf-8 -*-
"""
This file contains general helper functions for logging, error handling and
other utilities.
It should not contain application specific helpers, mathemcatical / statistical
methods, database querying or any other helper functions.
"""

import logging
import requests

def setup_logging(logger_name, log_file, level=logging.INFO,
                  print_level=False, mode='a'):
    """Sets up a logger with format ``timestamp: message``

    This method is a simple wrapper to create a log file using Python's standard
    logging library. If ``logger_name`` already exists, the same logger will be
    used, else a new logger is created.

    Args:
        **logger_name**: Name of the logger object\n
        **log_file**: The path to the log file\n
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
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def register_model(app_name, app_path):
    """Registers or updates a model with the model deployment framework

    Args:
        **app_name**: The name of the application
        **app_path**: The location of the application's Python module, relative
        to the working directory of the server

    Return:
        A `requests.Response` object
    """
    return requests.put('localhost:8080/register', json={'app_name': app_name, 'app_path': app_path})
