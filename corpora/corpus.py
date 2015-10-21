import gensim
import pickle
import pandas as pd
import nltk
import pattern.nl as nlp
import sys
import os
import progressbar
from time import time

class Corpus(object):
    def __init__(self, documents = [], metadata = [], nlp_tags=None, exclude_words=None):
        self.documents = documents
        self.metadata = metadata
        if nlp_tags is None:
            self._nlp_tags = None
        else:
            self._nlp_tags = frozenset(nlp_tags)
        if exclude_words is None:
            self._exclude_words = None
        else:
            self._exclude_words = frozenset(exclude_words)

    def add_file(self, fp, metadata={}):
        self.add_text(fp.read(), metadata)

    def add_text(self, text, metadata={}):
        tokens = self.tokenize(text)
        self.documents.append(tokens)
        self.metadata.append(metadata)

    @property
    def nlp_tags(self):
        if self._nlp_tags is None:
            self._nlp_tags = frozenset([
                # None, u'(', u')', u',', u'.', u'<notranslation>damesbladen', # extras
                # u'CC', # conjunction
                # u'CD', # cardinal (numbers)
                # u'DT', # determiner (de, het)
                u'FW', # foreign word
                # u'IN', #conjunction
                u'JJ', # adjectives -- # u'JJR', u'JJS',
                # u'MD', # Modal verb
                u'NN', u'NNP', u'NNPS', u'NNS', # Nouns
                # u'PRP', # Pronouns -- # u'PRP$',
                u'RB', # adverb
                u'RP', # adverb
                # u'SYM', # Symbol
                # u'TO', # infinitival to
                # u'UH', # interjection
                u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ', # Verb forms
                ])
        return self._nlp_tags

    @property
    def exclude_words(self):
        if self._exclude_words is None:
            stopwords = nltk.corpus.stopwords.words('english')
            stopwords += ['', '.', ',', '?', '(', ')', ',', ':', "'",
                          u'``', u"''", ';','-','!','%','&','...','=', '>',
                          '#', '_', '~', '+', '*', '/', '\\', '[', ']', '|']
            stopwords += [u'\u2019', u'\u2018', u'\u2013', u'\u2022',
                          u'\u2014', u'\uf02d', u'\u20ac', u'\u2026']
            self._exclude_words = frozenset(stopwords)
        return self._exclude_words

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

    def generate_dictionary(self):
        self.dic = gensim.corpora.Dictionary(self.documents)

    def generate_bag_of_words(self):
        if self.dic is None:
            self.generate_dictionary()

        self.corpus = [self.dic.doc2bow(tokens) for tokens in self.documents]


def load_vraagtekst_corpus(documents_filename):
    with open(documents_filename, 'r') as f:
        data_vraag = pickle.load(f)
    data_ppl = data_vraag[data_vraag['individu of groep']=='mijzelf']
    data_org = data_vraag[data_vraag['individu of groep']!='mijzelf']

    vraagTokens = data_vraag['SentToks'].tolist()

    dic = gensim.corpora.Dictionary(vraagTokens)
    corpus = [dic.doc2bow(text) for text in vraagTokens]
    return (dic, corpus, data_ppl)


def load_enron_corpus(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        total_size += len(files)

    print("reading {0} files".format(total_size))

    corpus = Corpus()

    bar = progressbar.ProgressBar(
        maxval=total_size,
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar('=', '[', ']'),
                 ' ', progressbar.widgets.ETA()])
    bar.start()

    for user in os.listdir(directory):
        for root, dirs, files in os.walk(os.path.join(directory, user)):
            mailbox = os.path.basename(root)
            for email in files:
                metadata = {'user': user, 'mailbox': mailbox, 'directory': root}
                with open(os.path.join(root, email)) as f:
                    corpus.add_file(f, metadata)
                if len(corpus.documents) % 10 == 0:
                    bar.update(len(corpus.documents))

    bar.finish()

    return corpus

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: corpus.py enron_directory enron_pickle_file.pkl")
        sys.exit(1)

    corpus = load_enron_corpus(sys.argv[1])
    print("Emails: {0}".format(len(corpus.documents)))
    with open(sys.argv[2], 'wb') as f:
        pickle.dump({'tokens': corpus.documents, 'metadata': corpus.metadata}, f)
