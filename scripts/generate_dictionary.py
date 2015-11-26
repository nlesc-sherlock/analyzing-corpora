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

import os
import argparse
from corpora.corpus import Corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="generate dictionary for a corpus")
    parser.add_argument(
        'parsed_document',
        help="python pickle file, containing tokens and metadata")
    parser.add_argument('dictionary', help="output dictionary")
    args = parser.parse_args()

    print("loading corpus")
    corpus = Corpus.load(args.parsed_document)
    print("generate dictionary")
    corpus.generate_dictionary()
    print("saving dictionary")
    corpus.save_dictionary(args.dictionary)

