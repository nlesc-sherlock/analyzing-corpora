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
import gensim
import pickle
import sys
import os
import progressbar
import itertools
import scipy
import scipy.io
import multiprocessing
import pandas as pd
import numpy as np
from .tokenizer import Tokenizer, filter_email


class Corpus(object):
    """ Stores a corpus along with its dictionary. """

    def __init__(self, documents=None, metadata=None, dictionary=None):
        if documents is None:
            self.documents = []
        else:
            self.documents = documents

        self._extra_metadata = []
        self._metadata = pd.DataFrame(metadata)
        # reindex to match documents
        self._metadata.index = np.arange(len(self._metadata))

        if dictionary is None:
            self._dic = gensim.corpora.Dictionary(self.documents)
        else:
            self._dic = dictionary

        self._csr_matrix = None

    @property
    def num_samples(self):
        return len(self.documents)

    @property
    def num_features(self):
        return len(self.dic)

    @property
    def dic(self):
        return self._dic

    @dic.setter
    def dic(self, new_dic):
        self._dic = new_dic
        self._reset_index()

    @property
    def metadata(self):
        if len(self._extra_metadata) > 0:
            frames = [self._metadata, pd.DataFrame(self._extra_metadata)]
            self._metadata = pd.concat(frames, ignore_index=True)
            self._extra_metadata = []
        return self._metadata

    def word(self, i):
        """ Returns the word with id in the dictionary. """
        return self.dic[i]

    def add_document(self, tokens, metadata={}, update_dictionary=True):
        """ Add a document to the corpus as a list of tokens """
        self.documents.append(tokens)
        self._extra_metadata.append(metadata)
        if update_dictionary:
            self.dic.add_documents([tokens])
            self._reset_index()

    def generate_dictionary(self):
        """ Generate the dictionary from the current corpus """
        self.dic = gensim.corpora.Dictionary(self.documents)

    # dictionary changed; index is no longer valid
    def _reset_index(self):
        self._csr_matrix = None

    def sparse_matrix(self):
        """ A sparse matrix M, with m_ij as the number of times word i occurs
        in document j. """
        if self._csr_matrix is None:
            doc_ids = []
            word_ids = []
            counts = []
            for doc_id, tokens in enumerate(self.documents):
                bow = self.dic.doc2bow(tokens)
                if len(bow) > 0:
                    new_word_ids, new_counts = zip(*bow)

                    doc_ids += list(itertools.repeat(doc_id, len(bow)))
                    word_ids += new_word_ids
                    counts += new_counts

            self._csr_matrix = scipy.sparse.csr_matrix(
                (counts, (doc_ids, word_ids)),
                shape=(self.num_samples, self.num_features))

        return self._csr_matrix

    def merge(self, other):
        """ Merge current corpus with another. This modifies the current
        corpus. Documents from the other corpus will come after the documents
        of the current one."""
        self.documents = self.documents + other.documents
        frames = [self.metadata, other.metadata]
        self._metadata = pd.concat(frames, ignore_index=True)
        self.dic.merge_with(other.dic)
        self._reset_index()

    def with_index(self, idx):
        """ Creates a new corpus with only the document at index idx, but with
        the same dictionary. """
        return Corpus(documents=[self.documents[idx]],
                      metadata=self.metadata[idx:idx+1],
                      dictionary=self.dic)

    def with_mask(self, mask):
        """ Creates a new corpus with all documents in the mask array. The
        mask array may be an integer index or bool mask."""
        new_docs = [d for i, d in enumerate(self.documents) if mask[i]]
        return Corpus(documents=new_docs,
                      metadata=self.metadata[mask],
                      dictionary=self.dic)

    def with_property(self, name, value):
        """ Creates a new corpus with all documents for which the metadata
        property name equals value."""
        return self.with_mask(self.metadata[name] == value)

    def with_tokens(self, tokens, metadata=None):
        """ Create a new corpus with the same dictionary but with a single
        list of tokens. """
        return Corpus(documents=[tokens],
                      metadata=metadata,
                      dictionary=self.dic)

    def filter_extremes(self, *args, **kwargs):
        """ Filters extreme occurrance values from the dictionary. See
        gensim.dictionary.Dictionary.filter_extremes for the arguments. Resets
        the indexes and invalidates the sparse matrix."""
        self.dic.filter_extremes(*args, **kwargs)
        self._reset_index()

    def save(self, documents_file, dictionary_file=None,
             metadata_filename=None):
        try:
            pickle.dump(self.documents, documents_file)
        except AttributeError:
            with open(documents_file, 'wb') as f:
                pickle.dump(self.documents, f)

        if dictionary_file is not None:
            self.save_dictionary(dictionary_file)

        if metadata_filename is not None:
            self.metadata.to_hdf(metadata_filename, 'metadata',
                                 complevel=7, complib='zlib')

    def save_dictionary(self, filename_or_fp):
        self.dic.save(filename_or_fp)

    @classmethod
    def load(cls, documents_file=None, dictionary_file=None,
             metadata_filename=None):
        if documents_file is None and dictionary_file is None:
            raise ValueError("Need corpus or dictionary filename")

        try:
            docs = pickle.load(documents_file)
        except AttributeError:
            with open(documents_file, 'rb') as f:
                docs = pickle.load(f)
        except TypeError:
            docs = []

        if isinstance(docs, dict):
            metadata = docs['metadata']
            docs = docs['tokens']
        else:
            metadata = None

        try:
            dic = gensim.corpora.Dictionary.load(dictionary_file)
        except AttributeError:
            dic = None

        if metadata is None and metadata_filename is not None:
            metadata = pd.read_hdf(metadata_filename, 'metadata')

        return cls(documents=docs, metadata=metadata,
                   dictionary=dic)

    def load_dictionary(self, filename_or_fp):
        self.dic = gensim.corpora.Dictionary.load(filename_or_fp)


def load_files(user, path, files, result_queue=None):
    corpus = Corpus()
    tokenizer = Tokenizer(filters=[filter_email])
    mailbox = os.path.basename(path)
    for email in files:
        metadata = {'user': user, 'mailbox': mailbox, 'directory': path}
        tokens = tokenizer.tokenize_file(os.path.join(path, email))
        corpus.add_document(tokens, metadata, update_dictionary=False)
    if result_queue is None:
        return corpus
    else:
        result_queue.put(corpus)


def load_vraagtekst_corpus(documents_filename):
    with open(documents_filename, 'r') as f:
        data_vraag = pickle.load(f)

    metadata_columns = data_vraag.columns.difference(['SentToks'])
    return Corpus(documents=data_vraag['SentToks'].tolist(),
                  metadata=data_vraag[metadata_columns])


def count_files(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        total_size += len(files)

    return total_size


def load_enron_corpus(directory):
    total_size = count_files(directory)
    print("reading {0} files".format(total_size))
    corpus = Corpus()

    bar = progressbar.ProgressBar(
        maxval=total_size,
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'),
                 ' ', progressbar.widgets.ETA()])
    bar.start()

    for user in os.listdir(directory):
        for path, dirs, files in os.walk(os.path.join(directory, user)):
            if len(files) > 0:
                corpus.merge(load_files(user, path, files))
                bar.update(len(corpus.documents))

    bar.finish()

    return corpus


def load_enron_corpus_mp(directory, num_processes=2):
    total_size = count_files(directory)
    print("reading {0} files".format(total_size))
    corpus = Corpus()

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    pool = multiprocessing.Pool(processes=num_processes)

    bar = progressbar.ProgressBar(
        maxval=total_size,
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'),
                 ' ', progressbar.widgets.ETA()])

    print("assigning tasks")
    bar.start()

    files_added = 0
    results_added = 0
    for user in os.listdir(directory):
        for path, dirs, files in os.walk(os.path.join(directory, user)):
            if len(files) > 0:
                pool.apply_async(load_files, [user, path, files, result_queue])
                files_added += len(files)
                results_added += 1
                bar.update(files_added)
    bar.finish()

    print("reading results")
    bar = progressbar.ProgressBar(
        maxval=total_size,
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'),
                 ' ', progressbar.widgets.ETA()])
    bar.start()

    for i in range(results_added):
        corpus.merge(result_queue.get())
        bar.update(len(corpus.documents))

    bar.finish()

    print("generating dictionary")
    corpus.generate_dictionary()

    pool.close()
    pool.join()

    return corpus

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Usage: corpus.py enron_directory enron_pickle_file.pkl")
    #     sys.exit(1)

    corpus = load_enron_corpus_mp(sys.argv[1])
    print("Emails: {0}".format(len(corpus.documents)))

    corpus.save(sys.argv[2], dictionary_filename=sys.argv[3])

    # corpus.generate_bag_of_words()
    # corpus.generate_corpus_matrix()
    # print("nSamples (docs) : {0}".format(len(corpus.corpus)))
    # print("nFeatures(words): {0}".format(len(corpus.dic)))
    # #corpus.scikit_lda(n_topics=5)
    # lda = scikit_lda(corpus.csr_matrix, n_topics = 5)
    # topicWords, topicWeightedWords = topic_words(lda, corpus.dic)
    # print("topicWords:",topicWords)
    # print("topicWeightedWords:", topicWeightedWords)
