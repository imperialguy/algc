# -*- coding: utf-8 -*-
"""
This file contains general helper functions for logging, error handling and
other utilities.
It should not contain application specific helpers, mathemcatical / statistical
methods, database querying or any other helper functions.
"""
from functools import update_wrapper
from json import JSONEncoder
import logging
import requests
import numpy
import time
import sys
import json


def _byteify(data, ignore_dicts=False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(
                value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )


def time_it(function):
    def timer(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        object_reference = args[0]
        time_expression = '{0} took {1} seconds'.format(
            function.__name__, round(end - start, 2))
        if getattr(object_reference, 'logger'):
            object_reference.logger.debug(time_expression)
        else:
            print(time_expression)
        return result
    return timer


class ALGCJSONEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(ALGCJSONEncoder, self).default(obj)


class mproperty(object):

    """ Use as a class method decorator.  It operates almost exactly like the
    Python ``@property`` decorator, but it puts the result of the method it
    decorates into the instance dict after the first call, effectively
    replacing the function it decorates with an instance variable. In other
    words, it is a `memoized property`

    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)

        return value


class NestedDict(dict):

    """ Nested Dictionary class

    """

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class AttributeDict(dict):

    """ Provides dictionary like object with values also accessible
     by attribute

    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def setup_logging(logger_name, log_file=None, level=logging.INFO,
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
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(
            '[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

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
    return requests.put('localhost:8080/register',
                        json={'app_name': app_name, 'app_path': app_path},
                        headers={'App-Name': 'falcon_server',
                                 'Authorization': 'Token BADA55'}
                        )
