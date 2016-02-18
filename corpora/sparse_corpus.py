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

from __future__ import print_function
import pandas as pd
import numpy as np
import itertools
from .corpus import Corpus, AbstractCorpus
from scipy.sparse import csr_matrix


class SparseCorpus(AbstractCorpus):

    """ Stores a sparse matrix along with its static dictionary. """

    def __init__(self, sparse_matrix, dictionary, metadata=None):
        super(SparseCorpus, self).__init__(metadata=metadata,
                                           dictionary=dictionary,
                                           sparse_matrix=sparse_matrix)

    @property
    def num_samples(self):
        return self._csr_matrix.shape[0]

    def with_indexed_mask(self, index):
        """ Creates a new corpus with all documents in the mask array. The
        mask array may be an integer index or bool mask."""
        return SparseCorpus(sparse_matrix=self._csr_matrix[index],
                            metadata=self.metadata[index],
                            dictionary=self.dic)

    def with_tokens(self, tokens, metadata=None):
        """ Create a new corpus with the same dictionary but with a single
        list of tokens. """
        return SparseCorpus(
            sparse_matrix=Corpus(
                documents=[tokens],
                dictionary=self.dic
            ).sparse_matrix(),
            metadata=metadata,
            dictionary=self.dic)

    def to_corpus(self, **kwargs):
        # n empty documents
        docs = list(itertools.repeat([], self.num_samples))

        # a COO matrix is easier to iterate over
        cx = self._csr_matrix.tocoo()
        # create document contents as lists of iterators
        for doc_id, word_id, count in itertools.izip(cx.row, cx.col, cx.data):
            docs[doc_id].append(itertools.repeat(self.dic[word_id], count))
        # expand the iterators
        docs = [list(itertools.chain(*doc)) for doc in docs]

        return Corpus(documents=docs, dictionary=self.dic,
                      metadata=self.metadata, **kwargs)

    def _reset_index(self):
        raise NotImplementedError("Sparse Corpus index cannot be reset")

    @classmethod
    def from_corpus(cls, corpus):
        return cls(sparse_matrix=corpus.sparse_matrix(), dictionary=corpus.dic)

    def save(self, sparse_matrix_file, dictionary_file=None,
             metadata_filename=None):
        mat = self.sparse_matrix()
        np.savez(sparse_matrix_file, data=mat.data, indices=mat.indices,
                 indptr=mat.indptr, shape=mat.shape)
        super(SparseCorpus, self).save(dictionary_file=dictionary_file,
                                       metadata_filename=metadata_filename)

    @classmethod
    def load(cls, sparse_matrix_file, dictionary_file, metadata_filename=None):
        loader = np.load(sparse_matrix_file)
        sparse_matrix = csr_matrix(
            (loader['data'], loader['indices'], loader['indptr']),
            shape=loader['shape'])

        dic = AbstractCorpus.load_dictionary(dictionary_file)
        if metadata_filename is not None:
            metadata = pd.read_hdf(metadata_filename, 'metadata')
        else:
            metadata = None

        return cls(sparse_matrix=sparse_matrix, dictionary=dic,
                   metadata=metadata)


def load_sparse_corpus(sparse_matrix_file=None, documents_file=None,
                       dictionary_file=None, metadata_filename=None):
    if sparse_matrix_file is not None:
        return SparseCorpus.load(
            sparse_matrix_file=sparse_matrix_file,
            dictionary_file=dictionary_file,
            metadata_filename=metadata_filename)
    else:
        return SparseCorpus.from_corpus(
            Corpus.load(
                documents_file=documents_file,
                dictionary_file=dictionary_file,
                metadata_filename=metadata_filename))
