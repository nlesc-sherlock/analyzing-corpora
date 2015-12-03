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

from .corpus import (
    Corpus,
    load_vraagtekst_corpus,
    load_enron_corpus_mp,
)
from .scikit import (
    ScikitLda,
    topics_by_discrete_property,
    topics_by_integer_property,
    plot_wordcloud_with_property,
)
from .sparse_corpus import (
    SparseCorpus,
    load_sparse_corpus,
)
from .tokenizer import (
    Tokenizer,
    filter_email,
)

__all__ = [
    'Corpus',
    'load_vraagtekst_corpus',
    'load_enron_corpus',
    'load_enron_corpus_mp',
    'ScikitLda',
    'topics_by_discrete_property',
    'topics_by_integer_property',
    'plot_wordcloud_with_property',
    'SparseCorpus',
    'load_sparse_corpus',
    'Tokenizer',
    'filter_email',
]
