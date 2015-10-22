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
import pattern.nl as nlp
import sys
import os
import progressbar
from scipy.sparse import csr_matrix
import re
import multiprocessing
import pandas as pd


class Corpus(object):
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

        if metadata is None:
            self.metadata = []
        else:
            self.metadata = metadata

        self._metadata_frame = None

        if dictionary is None:
            self.dic = gensim.corpora.Dictionary(self.documents)
        else:
            self.dic = dictionary

        self._corpus = None
        self._csr_matrix = None

        if nlp_tags is None:
            self.nlp_tags = Corpus.standard_nlp_tags
        else:
            self.nlp_tags = frozenset(nlp_tags)
        if exclude_words is None:
            self.exclude_words = Corpus.standard_stopwords
        else:
            self.exclude_words = frozenset(exclude_words)

    def add_file(self, fp, metadata={}, update_dictionary=True):
        self.add_text(fp.read(), metadata)

    def add_text(self, text, metadata={}, update_dictionary=True):
        tokens = self.tokenize(text)
        self.documents.append(tokens)
        self.metadata.append(metadata)
        self._metadata_frame = None
        if update_dictionary:
            self.dic.add_documents([tokens])
            self._reset_index()

    @property
    def num_samples(self):
        return len(self.documents)

    @property
    def num_features(self):
        return len(self.dic)

    def tokenize(self, text, nlp_tags=None, exclude_words=None):
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

    def load_dictionary(self, filename):
        self.dic = gensim.corpora.Dictionary.load(filename)
        self._reset_index()

    def generate_dictionary(self):
        self.dic = gensim.corpora.Dictionary(self.documents)
        self._reset_index()

    def indexed_corpus(self):
        if self._corpus is None:
            self._corpus = [self.dic.doc2bow(tokens)
                            for tokens in self.documents]
        return self._corpus

    def _reset_index(self):
        self._corpus = None
        self._csr_matrix = None

    def sparse_matrix(self):
        if self._csr_matrix is None:
            index = self.indexed_corpus()

            data = []
            row = []
            col = []
            for n, doc in enumerate(index):
                for w, c in doc:
                    col.append(n)
                    row.append(w)
                    data.append(c)

            self._csr_matrix = csr_matrix(
                (data, (col, row)),
                shape=(self.num_samples, self.num_features))

        return self._csr_matrix

    def save_dictionary(self, filename):
        self.dic.save(filename)

    def merge(self, other):
        self.documents = self.documents + other.documents
        self.metadata = self.metadata + other.metadata
        self._metadata_frame = None
        self.dic.merge_with(other.dic)
        self._reset_index()

    @property
    def metadata_frame(self):
        if self._metadata_frame is None:
            self._metadata_frame = pd.DataFrame.from_dict(self.metadata)
        return self._metadata_frame

    def with_index(self, idx):
        return Corpus(documents=[self.documents[idx]],
                      metadata=[self.metadata[idx]],
                      dictionary=self.dic,
                      nlp_tags=self.nlp_tags,
                      exclude_words=self.exclude_words)

    def with_mask(self, mask):
        new_docs = [d for i, d in enumerate(self.documents) if mask[i]]
        return Corpus(documents=new_docs,
                      metadata=self.metadata_frame[mask].to_dict('records'),
                      dictionary=self.dic,
                      nlp_tags=self.nlp_tags,
                      exclude_words=self.exclude_words)

    def with_property(self, name, value):
        return self.with_mask(self.metadata_frame[name] == value)

    def word(self, i):
        return self.dic[i]

    def filter_extremes(self, *args, **kwargs):
        self.dic.filter_extremes(*args, **kwargs)
        self._reset_index()

    def with_tokens(self, tokens):
        return Corpus(documents=[tokens],
                      metadata=None,
                      dictionary=self.dic,
                      nlp_tags=self.nlp_tags,
                      exclude_words=self.exclude_words)


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
    text = forward_pattern.sub('\n', text)
    text = html_patten.sub(' ', text)
    text = mime_pattern.sub(' ', text)
    return dot_pattern.sub('. ', text)


def load_vraagtekst_corpus(documents_filename):
    with open(documents_filename, 'r') as f:
        data_vraag = pickle.load(f)

    metadata_columns = data_vraag.columns.difference(['SentToks'])
    metadata = data_vraag.ix[:, metadata_columns].to_dict('records')
    return Corpus(documents=data_vraag['SentToks'].tolist(),
                  metadata=metadata)


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

    pool.close()
    pool.join()

    return corpus

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Usage: corpus.py enron_directory enron_pickle_file.pkl")
    #     sys.exit(1)

    corpus = load_enron_corpus_mp(sys.argv[1])
    print("Emails: {0}".format(len(corpus.documents)))

    with open(sys.argv[2], 'wb') as f:
        pickle.dump(
            {'tokens': corpus.documents, 'metadata': corpus.metadata}, f)
    print("saving dictionary")
    corpus.save_dictionary(sys.argv[3])

    # corpus.generate_bag_of_words()
    # corpus.generate_corpus_matrix()
    # print("nSamples (docs) : {0}".format(len(corpus.corpus)))
    # print("nFeatures(words): {0}".format(len(corpus.dic)))
    # #corpus.scikit_lda(n_topics=5)
    # lda = scikit_lda(corpus.csr_matrix, n_topics = 5)
    # topicWords, topicWeightedWords = topic_words(lda, corpus.dic)
    # print("topicWords:",topicWords)
    # print("topicWeightedWords:", topicWeightedWords)
