#!/usr/bin/env python
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

from __future__ import print_function
import argparse
from corpora import corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="parse and clean an email folder with headers already"
                    " removed")
    parser.add_argument('email_folder')
    parser.add_argument(
        'corpus',
        help="python pickle file, containing tokens")
    parser.add_argument('-d', '--dictionary', default=None)
    parser.add_argument('-m', '--metadata', default=None)
    parser.add_argument('-p', '--processes', default=2, type=int)
    parser.add_argument(
        '-f', '--format', default='python', help='Options: python, mm or scala')
    args = parser.parse_args()

    c = corpus.load_enron_corpus_mp(args.email_folder,
                                    num_processes=int(args.processes))

    print("storing python pickle file")
    if args.format == 'python':
        c.save(documents_file=args.corpus, dictionary_file=args.dictionary,
               metadata_filename=args.metadata)
    elif args.format == 'mm':
        c.save_mm(documents_file=args.corpus, dictionary_file=args.dictionary,
                  metadata_filename=args.metadata)
    elif args.format == 'scala':
        c.save_scala(documents_file=args.corpus, dictionary_file=args.dictionary,
                     metadata_filename=args.metadata)
    else:
        print('Unknown save format: {}'.format(args.format))
