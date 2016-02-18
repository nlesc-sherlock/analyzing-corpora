#!/usr/bin/env python
#
# Copyright 2015 Netherlands eScience Center <info@esciencecenter.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
description:    Utilities used in Sherlock
license:        APACHE 2.0
author:         Ronald van Haren, NLeSC (r.vanharen@esciencecenter.nl)
'''

import logging
import os

# define global LOG variables
DEFAULT_LOG_LEVEL = 'debug'
LOG_LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
LOG_LEVELS_LIST = LOG_LEVELS.keys()
#LOG_FORMAT = '%(asctime)-15s %(message)s'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATE_FORMAT = "%Y/%m/%d/%H:%M:%S"
logger = None


def start_logging(filename, level=DEFAULT_LOG_LEVEL):
    '''
    Start logging with given filename and level.
    '''
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)
    logger.setLevel(LOG_LEVELS[level])
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _create_directory(path):
    '''
    Create a directory if it does not exist yet
    '''
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:  # directory already exists, no problem
            raise  # re-raise exception if a different error occured
