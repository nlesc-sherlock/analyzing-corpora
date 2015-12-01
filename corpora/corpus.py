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
from abc import ABCMeta, abstractmethod
from .tokenizer import Tokenizer, filter_email


class AbstractCorpus(object):
    __metaclass__ = ABCMeta

    def __init__(self, metadata=None, dictionary=None, sparse_matrix=None):
        self._metadata = pd.DataFrame(metadata)
        # reindex to match documents
        self._metadata.index = np.arange(len(self._metadata))

        self._dic = dictionary
        self._csr_matrix = sparse_matrix

    @property
    def num_features(self):
        return len(self.dic)

    @property
    def dic(self):
        return self._dic

    @dic.setter
    def dic(self, new_dic):
        self._reset_index()
        self._dic = new_dic

    @property
    def metadata(self):
        return self._metadata

    def _reset_index(self):
        self._csr_matrix = None

    def word(self, i):
        """ Returns the word with id in the dictionary. """
        return self.dic[i]

    def sparse_matrix(self):
        return self._csr_matrix

    def with_mask(self, mask):
        """ Creates a new corpus with all documents in the mask array. The
        mask array may be an integer index or bool mask."""
        index = np.asarray(mask)
        if index.dtype == bool:
            index = np.where(index)[0]
        return self.with_indexed_mask(index)

    @abstractmethod
    def with_indexed_mask(self, index):
        raise NotImplementedError

    def with_index(self, idx):
        """ Creates a new corpus with only the document at index idx, but with
        the same dictionary. """
        return self.with_mask([idx])

    def with_property(self, name, value):
        """ Creates a new corpus with all documents for which the metadata
        property name equals value."""
        return self.with_mask(self.metadata[name] == value)

    @abstractmethod
    def with_tokens(self, tokens, metadata=None):
        """ Create a new corpus with the same dictionary but with a single
        list of tokens. """
        raise NotImplementedError

    def save_dictionary(self, dictionary_filename):
        self.dic.save(dictionary_filename)

    def save(self, dictionary_file=None,
             metadata_filename=None):
        if dictionary_file is not None:
            self.save_dictionary(dictionary_file)

        if metadata_filename is not None:
            self.metadata.to_hdf(metadata_filename, 'metadata',
                                 complevel=7, complib='zlib')

    def save_csv(self, dictionary_file=None, metadata_filename=None):
        if dictionary_file is not None:
            self.dic.save_as_text(dictionary_file)
        if metadata_filename is not None:
            self.metadata.to_csv(metadata_filename)

class Corpus(AbstractCorpus):
    """ Stores a corpus along with its dictionary. """

    def __init__(self, documents=None, metadata=None, dictionary=None):
        if documents is None:
            # cannot move this to parameter default, because then the list
            # will be shared between Corpus objects.
            self.documents = []
        else:
            self.documents = documents

        if dictionary is None:
            dictionary = gensim.corpora.Dictionary(self.documents)

        super(Corpus, self).__init__(metadata=metadata, dictionary=dictionary)

        self._extra_metadata = []

    @property
    def num_samples(self):
        return len(self.documents)

    @property
    def metadata(self):
        if len(self._extra_metadata) > 0:
            frames = [self._metadata, pd.DataFrame(self._extra_metadata)]
            self._metadata = pd.concat(frames, ignore_index=True)
            self._extra_metadata = []
        return super(Corpus, self).metadata

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

        return super(Corpus, self).sparse_matrix()

    def merge(self, other):
        """ Merge current corpus with another. This modifies the current
        corpus. Documents from the other corpus will come after the documents
        of the current one."""
        self.documents = self.documents + other.documents
        frames = [self.metadata, other.metadata]
        self._metadata = pd.concat(frames, ignore_index=True)
        self.dic.merge_with(other.dic)
        self._reset_index()

    def with_indexed_mask(self, index):
        return Corpus(documents=[self.documents[i] for i in index],
                      metadata=self.metadata[index],
                      dictionary=self.dic)

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

        super(Corpus, self).save(dictionary_file=dictionary_file,
                                 metadata_filename=metadata_filename)

    def save_csv(self, documents_file, dictionary_file=None,
                 metadata_filename=None):
            with open(documents_file, 'w') as fout:
                for docId,doc in enumerate(self.documents):
                    print('{} {}'.format(docId, doc))
                    bow = self.dic.doc2bow(doc)
                    for wordId,count in bow:
                        fout.write('{} {} {}\n'.format(docId,wordId,count))

            super(Corpus, self).save_csv(dictionary_file=dictionary_file,
                                     metadata_filename=metadata_filename)

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

    def load_dictionary(self, dictionary_file):
        self.dic = gensim.corpora.Dictionary.load(dictionary_file)


def load_files(user, path, files, result_queue=None):
    corpus = Corpus()
    tokenizer = Tokenizer(filters=[filter_email])
    mailbox = os.path.basename(path)
    for email in files:
        metadata = {'user': user, 'mailbox': mailbox, 'directory': path}
        tokens = tokenizer.tokenize_file(os.path.join(path, email))
        # corpus.add_document(tokens, metadata, update_dictionary=True)
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
