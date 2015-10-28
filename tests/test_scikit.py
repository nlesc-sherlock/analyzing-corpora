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

from nose.tools import assert_equals, assert_false
import numpy as np
from numpy.testing import assert_array_less
from test_mock import mock_corpus
from corpora import ScikitLda


def test_scikit_init():
    c, docs = mock_corpus()
    ScikitLda(corpus=c)


def test_scikit_fit():
    c, docs = mock_corpus()
    lda = ScikitLda(corpus=c, n_topics=2)
    lda.fit()
    assert_equals((len(docs), 2), lda.weights.shape)
    # 0 <= weight <= 1
    assert_array_less(-0.001, lda.weights)
    assert_array_less(lda.weights, 1.001)


def test_scikit_partial_fit():
    c, docs = mock_corpus()
    n_topics = 3
    lda = ScikitLda(corpus=c, n_topics=n_topics)
    lda.fit()

    plda = ScikitLda(corpus=c, n_topics=n_topics)
    plda.partial_fit(c.with_index(0))
    assert_equals(lda.weights.shape, plda.weights.shape)

    first_fit = np.array(plda.weights)
    # only a partial fit will not come close to a full fit
    assert_false(np.allclose(first_fit, lda.weights))

    plda.partial_fit(c.with_index(1))
    assert_equals(lda.weights.shape, plda.weights.shape)
    second_fit = np.array(plda.weights)
    # a second partial fit will differ from the first
    assert_false(np.allclose(second_fit, first_fit))
    # would like to test closeness of single fit with a number of partial
    # fits, but it will not work consistently
