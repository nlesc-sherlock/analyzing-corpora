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
import nltk
import pattern.en as nlp
import sys
import os
import progressbar
import scipy
import re
import multiprocessing
import pandas as pd
import numpy as np


class Corpus(object):
    """ Stores a corpus along with its dictionary. """

    standard_nlp_tags = frozenset([
        # None, u'(', u')', u',', u'.', u'<notranslation>damesbladen', # extras
        # u'CC', # conjunction
        # u'CD', # cardinal (numbers)
        # u'DT', # determiner (de, het)
        u'FW',  # foreign word
        # u'IN', #conjunction
        u'JJ',  # adjectives -- # u'JJR', u'JJS',
        # u'MD', # Modal verb
        u'NN', u'NNP', u'NNPS', u'NNS',  # Nouns
        # u'PRP', # Pronouns -- # u'PRP$',
        u'RB',  # adverb
        u'RP',  # adverb
        # u'SYM', # Symbol
        # u'TO', # infinitival to
        # u'UH', # interjection
        u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ',  # Verb forms
    ])
    standard_stopwords = frozenset(
        nltk.corpus.stopwords.words('english') +
        ['', '.', ',', '?', '(', ')', ',', ':', "'",
         u'``', u"''", ';', '-', '!', '%', '&', '...', '=', '>', '<',
         '#', '_', '~', '+', '*', '/', '\\', '[', ']', '|'
         u'\u2019', u'\u2018', u'\u2013', u'\u2022',
         u'\u2014', u'\uf02d', u'\u20ac', u'\u2026'])

    def __init__(self, documents=None, metadata=None, dictionary=None,
                 nlp_tags=None, exclude_words=None):
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

        if nlp_tags is None:
            self.nlp_tags = Corpus.standard_nlp_tags
        else:
            self.nlp_tags = frozenset(nlp_tags)
        if exclude_words is None:
            self.exclude_words = Corpus.standard_stopwords
        else:
            self.exclude_words = frozenset(exclude_words)

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

    def add_file(self, filename_or_fp, metadata={}, update_dictionary=True):
        """ Add a document to the corpus """
        try:
            text = filename_or_fp.read()
        except AttributeError:
            with open(filename_or_fp, 'rb') as fp:
                text = fp.read()

        self.add_text(text, metadata)

    def add_text(self, text, metadata={}, update_dictionary=True):
        """ Add a text to the corpus as a single string """
        tokens = self.tokenize(text)
        self.documents.append(tokens)
        self._extra_metadata.append(metadata)
        if update_dictionary:
            self.dic.add_documents([tokens])
            self._reset_index()

    def tokenize(self, text, nlp_tags=None, exclude_words=None):
        """
        Tokenize words in a text and return the relevant ones

        Parameters
        ----------
        text : str
            Text to tokenize.
        nlp_tags : list or set of str
            Natural language processing codes of word semantics to keep as
            relevant tokens when tokenizing. See Corpus.standard_nlp_tags for
            an example
        exclude_words : list or set of str
            Exact words and symbols to filter out.
        """
        if nlp_tags is None:
            nlp_tags = self.nlp_tags
        if exclude_words is None:
            exclude_words = self.exclude_words

        words = []
        for s in nlp.split(nlp.parse(text)):
            for word, tag in s.tagged:
                if tag in nlp_tags:
                    word = word.lower()
                    if word not in exclude_words:
                        words.append(word)

        return words

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
            doc_ids = np.array([], dtype=np.int8)
            word_count_tuples = np.ndarray(shape=(0, 2), dtype=np.int8)
            for n, tokens in enumerate(self.documents):
                bow = self.dic.doc2bow(tokens)
                # append to the big bag of words, we will transpose later
                # to separate the words and counts.
                word_count_tuples = np.append(word_count_tuples, bow, axis=0)
                # index the word_count_tuples with the doc id == row number
                doc_ids = np.append(doc_ids, np.repeat(n, len(bow)))

            word_ids = word_count_tuples.T[0]
            word_counts = word_count_tuples.T[1]

            self._csr_matrix = scipy.sparse.csr_matrix(
                (word_counts, (doc_ids, word_ids)),
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
                      dictionary=self.dic,
                      nlp_tags=self.nlp_tags,
                      exclude_words=self.exclude_words)

    def with_mask(self, mask):
        """ Creates a new corpus with all documents in the mask array. The
        mask array may be an integer index or bool mask."""
        new_docs = [d for i, d in enumerate(self.documents) if mask[i]]
        return Corpus(documents=new_docs,
                      metadata=self.metadata[mask],
                      dictionary=self.dic,
                      nlp_tags=self.nlp_tags,
                      exclude_words=self.exclude_words)

    def with_property(self, name, value):
        """ Creates a new corpus with all documents for which the metadata
        property name equals value."""
        return self.with_mask(self.metadata[name] == value)

    def with_tokens(self, tokens, metadata=None):
        """ Create a new corpus with the same dictionary but with a single
        list of tokens. """
        return Corpus(documents=[tokens],
                      metadata=metadata,
                      dictionary=self.dic,
                      nlp_tags=self.nlp_tags,
                      exclude_words=self.exclude_words)

    def filter_extremes(self, *args, **kwargs):
        """ Filters extreme occurrance values from the dictionary. See
        gensim.dictionary.Dictionary.filter_extremes for the arguments. Resets
        the indexes and invalidates the sparse matrix."""
        self.dic.filter_extremes(*args, **kwargs)
        self._reset_index()

    def save(self, filename_or_fp, dictionary_filename_or_fp=None):
        corpus_dict = {
            'tokens': self.documents,
            'metadata': self.metadata,
        }
        try:
            pickle.dump(corpus_dict, filename_or_fp)
        except AttributeError:
            with open(filename_or_fp, 'wb') as f:
                pickle.dump(corpus_dict, f)

        if dictionary_filename_or_fp is not None:
            self.save_dictionary(dictionary_filename_or_fp)

    def save_dictionary(self, filename_or_fp):
        self.dic.save(filename_or_fp)

    @classmethod
    def load(cls, filename_or_fp=None, dictionary_filename_or_fp=None):
        if filename_or_fp is None and dictionary_filename_or_fp is None:
            raise ValueError("Need corpus or dictionary filename")

        try:
            data = pickle.load(filename_or_fp)
        except AttributeError:
            with open(filename_or_fp, 'rb') as f:
                data = pickle.load(f)
        except TypeError:
            data = {'tokens': None, 'metadata': None}

        try:
            dic = gensim.corpora.Dictionary.load(dictionary_filename_or_fp)
        except TypeError:
            dic = None

        return cls(documents=data['tokens'], metadata=data['metadata'],
                   dictionary=dic)

    def load_dictionary(self, filename_or_fp):
        self.dic = gensim.corpora.Dictionary.load(filename_or_fp)


def load_files(user, path, files, result_queue=None):
    corpus = Corpus()
    mailbox = os.path.basename(path)
    for email in files:
        metadata = {'user': user, 'mailbox': mailbox, 'directory': path}
        with open(os.path.join(path, email)) as f:
            text = filter_email(f.read())
            corpus.add_text(text, metadata, update_dictionary=False)
    if result_queue is None:
        return corpus
    else:
        result_queue.put(corpus)


forward_pattern = re.compile('[\r\n]>[^\r\n]*[\r\n]')
html_patten = re.compile('<[^<]+?>')
mime_pattern = re.compile('=\d\d')
dot_pattern = re.compile('\.\.+')


def filter_email(text):
    """ Filters reply/forward text, html, mime encodings and dots from emails.
    """
    text = forward_pattern.sub('\n', text)
    text = html_patten.sub(' ', text)
    text = mime_pattern.sub(' ', text)
    return dot_pattern.sub('. ', text)


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
