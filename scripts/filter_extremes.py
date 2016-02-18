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

import os
import argparse
from corpora.corpus import Corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train and save the LDA parser with different number of "
                    "topics")
    parser.add_argument('input_dictionary')
    parser.add_argument('output_dictionary')
    parser.add_argument('-a', '--no-above', default=0.5)
    parser.add_argument('-b', '--no-below', default=5)
    parser.add_argument('-k', '--keep', default=1000000)
    args = parser.parse_args()

    print("loading dictionary")
    dic = AbstractCorpus.load_dictionary(args.input_dictionary)
    dic.filter_extremes(no_above=float(args.no_above),
                        no_below=int(args.no_below),
                        keep_n=int(args.keep))

    print("writing to file")
    dic.save_as_text(args.output_dictionary)
