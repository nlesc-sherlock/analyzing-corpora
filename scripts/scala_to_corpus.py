#!/usr/bin/env python
# Analyzing Corpora
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
        description="load files of scala file and convert it to a python "
        			"corpus using a dictionary")
    parser.add_argument(
        'scala_file',
        help="python pickle file, containing tokens and metadata")
    parser.add_argument('dictionary')
    parser.add_argument('corpus')
    args = parser.parse_args()

    print("loading scala_file")
    corpus = Corpus.load(
        scala_file=args.scala_file,
        dictionary_file=args.dictionary)

    print("writing corpus to file")
    corpus.save(documents_file=args.corpus)
