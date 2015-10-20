#!/usr/bin/env python2

import tarfile

class load_data:
  '''
  class to load txt data
  '''
  def __init__(self, filename):
    self.filename = filename
    tfile, members = self.get_archive_object_tar()
    self.read_files(tfile, members)

  def get_archive_object_tar(self):
    '''
    return tarfile object and its members
    '''
    tfile = tarfile.open(name = self.filename)
    members = tfile.getnames()
    return tfile, members
 
  def read_files(self, tfile, members):
    '''
    array with txt data from tarfile object
    '''
    self.data = [tfile.extractfile(member).read() for member in members if
                 tfile.extractfile(member) is not None]


def main():
  enron = load_data('enron_mail_clean.tar.gz')
  import pdb; pdb.set_trace()


if __name__ == "__main__":
  main()
