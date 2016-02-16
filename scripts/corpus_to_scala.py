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

import argparse
from corpora import Corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="load files of corpus and store scala corpus")
    parser.add_argument('-d', '--dictionary', default=None)
    parser.add_argument(
        'corpus',
        help="python pickle file, containing tokens and metadata")
    parser.add_argument('scala_file')
    args = parser.parse_args()

    print("loading corpus")
    corpus = Corpus.load(
        documents_file=args.corpus,
        dictionary_file=args.dictionary)

    print("writing scala csv file")
    corpus.save_scala(args.scala_file)
