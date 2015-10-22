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

import argparse
from corpora import corpus
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="parse and clean an email folder with headers already"
                    " removed")
    parser.add_argument('email_folder')
    parser.add_argument(
        'parsed_document',
        help="python pickle file, containing tokens and metadata")
    parser.add_argument('-d', '--dictionary', default=None)
    parser.add_argument('-p', '--processes', default=2)
    args = parser.parse_args()

    if args.processes == 1:
        c = corpus.load_enron_corpus(args.email_folder)
    else:
        c = corpus.load_enron_corpus_mp(args.email_folder,
                                        num_processes=int(args.processes))

    print("storing python pickle file")
    with open(args.parsed_document, 'wb') as f:
        pickle.dump({'tokens': c.documents, 'metadata': c.metadata}, f)

    if args.dictionary is not None:
        print "saving dictionary"
        c.save_dictionary(args.dictionary)
