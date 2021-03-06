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
import pandas as pd
from numpy.testing import assert_array_equal
import gensim
from corpora import Corpus
from test_mock import mock_corpus
import os
import tempfile


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


def test_sparse_matrix():
    c, docs = mock_corpus()
    matrix = c.sparse_matrix()
    assert_equals((2, 3), matrix.shape)
    assert_equals(2, matrix[0, 0])
    assert_equals(0, matrix[1, 0])
    assert_equals(1, matrix[1, 2])

    c.add_document([])
    matrix = c.sparse_matrix()
    assert_equals((3, 3), matrix.shape)
    assert_equals(2, matrix[0, 0])
    assert_equals(0, matrix[1, 0])
    assert_equals(1, matrix[1, 2])


def test_metadata():
    c, docs = mock_corpus()
    assert_true(isinstance(c.metadata, pd.DataFrame))
    assert_equals(2, len(c.metadata))
    assert_equals('this', c.metadata['user'][0])
    assert_equals(10, c.metadata['age'][0])
    assert_true(np.isnan(c.metadata['age'][1]))


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
    old_metadata = c.metadata.fillna(-1.0)

    c.merge(otherc)
    assert_equals(4, c.num_samples)
    assert_array_equal(docs + docs, c.documents)
    assert_equals(3, c.num_features)
    assert_equals((4, 3), c.sparse_matrix().shape)
    assert_equals(4, len(c.metadata))

    new_metadata = c.metadata[:2].fillna(-1.0)
    assert_array_equal(old_metadata, new_metadata)

    other_metadata = c.metadata[2:4].fillna(-1.0)
    # reset index to align with old metadata
    other_metadata.index = [0, 1]
    assert_array_equal(old_metadata, other_metadata)


def test_add():
    c, docs = mock_corpus()
    old_samples = c.num_samples
    old_features = c.num_features
    c.add_document(['new', 'words'])
    assert_equals(old_samples + 1, c.num_samples)
    assert_equals(old_features + 2, c.num_features)


def test_save_load():
    c, docs = mock_corpus()
    fd, filename = tempfile.mkstemp()
    dict_fd, dict_filename = tempfile.mkstemp()
    metadata_fd, metadata_filename = tempfile.mkstemp()
    try:
        f = None
        dict_f = None
        try:
            f = os.fdopen(fd, 'wb')
            dict_f = os.fdopen(dict_fd, 'wb')
            c.save(documents_file=f, dictionary_file=dict_f,
                   metadata_filename=metadata_filename)
        finally:
            if f is not None:
                f.close()
            if dict_f is not None:
                dict_f.close()

        new_c = Corpus.load(
            documents_file=filename,
            dictionary_file=dict_filename,
            metadata_filename=metadata_filename)
        assert_equals(c.documents, new_c.documents)
        assert_true(all(c.metadata == new_c.metadata))
        assert_equals(c.dic, new_c.dic)
    finally:
        os.remove(filename)
        os.remove(dict_filename)


def test_save_load_dictionary():
    c, docs = mock_corpus()
    dict_fd, dict_filename = tempfile.mkstemp()
    try:
        dict_f = None
        try:
            dict_f = os.fdopen(dict_fd, 'wb')
            c.save_dictionary(dict_f)
        finally:
            if dict_f is not None:
                dict_f.close()

        new_c = Corpus()
        new_c.load_dictionary(dict_filename)
        assert_equals(c.dic, new_c.dic)

    finally:
        os.remove(dict_filename)
