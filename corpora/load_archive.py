#!/usr/bin/env python
# SIM-CITY client
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
        tfile = tarfile.open(name=self.filename)
        members = tfile.getnames()
        return tfile, members

    def read_files(self, tfile, members):
        '''
        array with txt data from tarfile object
        '''
        self.data = [tfile.extractfile(member).read() for member in members if
                     tfile.extractfile(member) is not None]


def main():
    load_data('enron_mail_clean.tar.gz')
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
