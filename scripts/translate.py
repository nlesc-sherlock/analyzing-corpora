#!/usr/bin/env python

'''
description:    Sherlock translate e-mails using online translation service
license:        APACHE 2.0
author:         Ronald van Haren, NLeSC (r.vanharen@esciencecenter.nl)

note:           A file called 'yandex_key' should be present in the CWD
                containing your personal yandex api key
'''

import os
import sqlite3
import logging
import numpy
import random
from yandex_translate import YandexTranslate
from yandex_translate import YandexTranslateException
import utils
import argparse
import sys


class yandex_skipped(Exception):
  def __init__(self):
    Exception.__init__(self, "unable to translate file using yandex") 


class translate:
  def __init__(self, input_dir, dest_dir):
    global logger
    logger = utils.start_logging('translate.log')  # create log file
    self.input_dir = input_dir
    self.dest_dir = dest_dir
    self.database = 'translate.db'  # hardcode database name for now
    try:
        with open(self.database) as file:
            pass  # file exists and is readable, nothing else to do
    except IOError as e:
      # file does not exist OR no read permissions
      # create database if not existing
      self._create_database()
    # read yandex api key from file
    self._read_yandex_api_key('yandex_key')  # TODO: hardcode for now
    self._yandex_connect(self.api_key)


  def _create_database(self):
    '''
    create database and add all files to it
    '''
    self._new_database()
    self._create_list_of_files()
    self._add_files_to_database()
    self._close_connection()

  def _create_list_of_files(self):
    '''
    create a list of files in a directory structure: self.files
    list all files in the directory that is given as an argument to the
    function, searches all subdirectories recursively
    '''
    self.files = []
    for root, dirnames, filenames in os.walk(self.input_dir):
      for filename in filenames:
        self.files.append(os.path.join(root, filename))


  def _new_database(self):
    '''
    create and connect to a new sqlite database. 
    raise an error if there already is a database in place, asking the
    user to manually remove the database (for safety reasons)
    '''
    # TODO: remove next two lines after testing -> don't automatically remove
    if os.path.exists(self.database):
      os.remove(self.database)
    if os.path.exists(self.database):
      message = ('Database already exists, please remove manually: %s'
                 %self.database)
      logger.error(message)
      raise IOError(message)
    else:
      logger.info('Database not found, creating database %s' %self.database)
      try:
        self.connection = sqlite3.connect(
          self.database,
          detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
      except:
        message = 'Failed to create database: %s' %self.database
        logger.error(message)
        raise sqlite3.OperationalError(message) # re-raise error
      self._create_dbstructure()
      sqlite3.register_adapter(bool, int)
      sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))
      # tuples
      self.connection.row_factory = sqlite3.Row


  def _connect_to_database(self):
    '''
    check if database exists and try to connect to the database
    '''
    #utils.check_file_exists(self.database)  # check if database exists
    try:
      logger.debug('Connecting to database: %s' %self.database)
      self.connection = sqlite3.connect(
        self.database,
        detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
      self.cursor = self.connection.cursor()
    except:
      message = 'Database %s exists, but failed to connect' %self.database
      logger.error(message)
      raise


  def _close_connection(self):
    '''
    close connection to database
    '''
    try:
      logger.debug('Closing database connection')
      self.connection.close()
    except NameError:
      logger.error(('Failed to close database connection, '
                    'variable does not exist'))


  def _create_dbstructure(self):
    '''
    create database structure
    '''
    #  self.connection.row_factory = sqlite3.Row  # use dict instread of tuple
    self.cursor = self.connection.cursor()
    # create table
    self.cursor.execute(
      '''CREATE TABLE filelist
      (filename TEXT, processed BOOLEAN, skipped BOOLEAN)''')


  def _add_files_to_database(self):
    '''
    add filenames to database table
    '''
    list_files = numpy.transpose(
      numpy.vstack((numpy.transpose(self.files),
                    numpy.transpose(numpy.zeros(len(self.files))),
                    numpy.transpose(numpy.zeros(len(self.files))))))
    logger.info('Adding filenames to database')
    # add all files to database
    self.cursor.executemany('''INSERT INTO filelist VALUES (?,?,?)''',
                        (list_files))
    # commit changes to database
    self.connection.commit()


  def _get_next_file(self):
    '''
    get all unprocessed files from database
    '''
    # connect to database
    self._connect_to_database()
    # query database for all unprocessed files
    self.cursor.execute("SELECT filename FROM filelist WHERE processed=(?)",
                        (False,))
    # pick a random unprocessed file
    self.next_file = random.choice(self.cursor.fetchall())[0]
    # close connection
    self._close_connection


  def _read_yandex_api_key(self, yandex_key):
    '''
    read yandex api key from file
    '''
    f = open(yandex_key, 'r')
    self.api_key = f.readline().strip('\n')
    f.close()


  def _yandex_connect(self, api_key):
    '''
    connect to yandex server with personal api_key
    '''
    self.translate = YandexTranslate(api_key)


  def read_next_file(self):
    '''
    read the next file to be processed
    '''
    f = open(self.next_file, 'r')
    self.text = f.read()
    f.close()

  def _translate_text(self, dest_lang='nl'):
    '''
    translate email to different language, default destination language is nl
    '''
    try:
      self.translated_text = self.translate.translate(self.text, dest_lang)
    except YandexTranslateException as e:
      if e[0]=='ERR_TEXT_TOO_LONG':
        logger.info('Unable to translate e-mail using yandex, ERR_TEXT_TOO_LONG')
        self._update_db_skipped()
        raise yandex_skipped

      else:
        logger.error('Error using Yandex Translation service, exiting...')
        raise


  def _save_translated_text(self):
    '''
    save translated text
    '''
    dest_file  = os.path.join(self.dest_dir, self.next_file.rsplit(
      self.input_dir)[1].strip('/'))
    dir_name = os.path.dirname(dest_file)
    utils._create_directory(dir_name)
    f = open(dest_file, 'w')
    f.write(self.translated_text['text'][0].encode('utf-8'))
    f.close()


  def _modify_database(self):
    '''
    modify database after file is processed
    '''
    self._connect_to_database()
    # update database processed
    self.cursor.execute("UPDATE filelist SET processed=(?) WHERE filename=(?)",
                        (True, self.next_file,))
    # commit changes to database
    self.connection.commit()
    # close connection
    self._close_connection


  def _update_db_skipped(self):
    '''
    modify database after file is processed
    '''
    self._connect_to_database()
    # update database skipped
    self.cursor.execute("UPDATE filelist SET skipped=(?) WHERE filename=(?)",
                        (True, self.next_file,))
    # update database processed
    self.cursor.execute("UPDATE filelist SET processed=(?) WHERE filename=(?)",
                        (True, self.next_file,))    
    # commit changes to database
    self.connection.commit()
    # close connection
    self._close_connection


  def _get_processed_files(self):
    '''
    get all unprocessed files from database
    '''
    # connect to database
    self._connect_to_database()
    # query database for all unprocessed files
    self.cursor.execute("SELECT filename FROM filelist WHERE processed=(?)",
                        (True,))
    # pick a random unprocessed file
    self.processed = self.cursor.fetchall()
    # close connection
    self._close_connection


  def process(self):
    '''
    process files
    stops automatically when yandex translation service returns an error
    yandex error codes are the following:
      error_codes = {
    401: "ERR_KEY_INVALID",
    402: "ERR_KEY_BLOCKED",
    403: "ERR_DAILY_REQ_LIMIT_EXCEEDED",
    404: "ERR_DAILY_CHAR_LIMIT_EXCEEDED",
    413: "ERR_TEXT_TOO_LONG",
    422: "ERR_UNPROCESSABLE_TEXT",
    501: "ERR_LANG_NOT_SUPPORTED",
    503: "ERR_SERVICE_NOT_AVAIBLE",
  }
    '''
    while True:
      self._get_next_file()
      if len(self.next_file)==0:
        logger.info('No more files to process, exiting...')
        sys.exit()
      else:
        logger.info('Processing file %s' %self.next_file)
      self.read_next_file()
      try:
        self._translate_text()
      except yandex_skipped:
        continue
      self._save_translated_text()
      self._modify_database()



if __name__=="__main__":
  parser = argparse.ArgumentParser(
    description='Translating a directory structure with text using yandex')
  parser.add_argument('--input_dir', help='Directory with input files', 
                      required=True)
  parser.add_argument('--output_dir',
                      help='Directory where output files should be saved',
                      required=True)
  args = parser.parse_args()

  # initialize
  tr = translate(args.input_dir, args.output_dir)
  # process files
  tr.process()
