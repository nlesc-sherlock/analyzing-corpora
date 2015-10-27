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

from nose.tools import assert_true, assert_equals
import numpy as np
from numpy.testing import assert_array_equal
import gensim
from corpora.corpus import Corpus, filter_email
from test_mock import mock_corpus


def test_init_corpus():
    c = Corpus()
    assert_equals([], c.documents)
    assert_equals(gensim.corpora.Dictionary(), c.dic)


def test_init_nonempty():
    c, docs = mock_corpus()

    assert_equals(docs, c.documents)
    assert_equals(gensim.corpora.Dictionary(docs), c.dic)
    assert_equals(3, c.num_features)
    assert_equals(2, c.num_samples)

    words = []
    for i in range(c.num_features):
        words.append(c.word(i))

    assert_equals(frozenset(words), frozenset(['a', 'la', 'ca']))


def test_indexed_corpus():
    c, docs = mock_corpus()
    indexed = c.indexed_corpus()
    assert_equals([[(0, 2), (1, 1)], [(2, 1)]], indexed)


def test_sparse_matrix():
    c, docs = mock_corpus()
    matrix = c.sparse_matrix()
    assert_equals((2, 3), matrix.shape)
    assert_equals(2, matrix[0, 0])
    assert_equals(0, matrix[1, 0])
    assert_equals(1, matrix[1, 2])


def test_metadata():
    c, docs = mock_corpus()
    assert_equals(2, len(c.metadata))
    assert_equals('this', c.metadata_frame['user'][0])
    assert_equals(10, c.metadata_frame['age'][0])
    assert_true(np.isnan(c.metadata_frame['age'][1]))


def test_index():
    c, docs = mock_corpus()
    newc = c.with_index(1)
    assert_equals(2, c.num_samples)
    assert_equals(1, newc.num_samples)
    # dictionary does not change
    assert_equals(3, c.num_features)
    assert_equals(3, newc.num_features)
    assert_array_equal(docs[1], newc.documents[0])


def test_mask():
    c, docs = mock_corpus()
    newc = c.with_mask([True, False])
    assert_equals(2, c.num_samples)
    assert_equals(1, newc.num_samples)
    # dictionary does not change
    assert_equals(3, c.num_features)
    assert_equals(3, newc.num_features)
    assert_array_equal(docs[0], newc.documents[0])

    newc = c.with_mask([False, True])
    assert_equals(1, newc.num_samples)
    assert_array_equal(docs[1], newc.documents[0])

    newc = c.with_mask([True, True])
    assert_equals(2, newc.num_samples)

    newc = c.with_mask([False, False])
    assert_equals(0, newc.num_samples)
    # dictionary does not change
    assert_equals(3, newc.num_features)


def test_merge():
    c, docs = mock_corpus()
    otherc, otherdocs = mock_corpus()

    c.merge(otherc)
    assert_equals(4, c.num_samples)
    assert_array_equal(docs + docs, c.documents)
    assert_equals(3, c.num_features)
    assert_equals((4, 3), c.sparse_matrix().shape)


def test_filter_email():
    email = """
Hi john
this <b>great</b>...<blink>opportunity!=20
> forwarded and so filtered
"""

    filtered = filter_email(email)
    assert_equals("\nHi john\nthis  great .  opportunity! \n", filtered)
