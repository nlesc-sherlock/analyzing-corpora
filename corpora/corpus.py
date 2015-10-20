import gensim
import pickle
import pandas as pd
import nltk
from pattern.nl import parse, split

def load_vraagtekst_corpus(documents_filename):
    with open(documents_filename, 'r') as f:
        data_vraag = pickle.load(f)
    data_ppl = data_vraag[data_vraag['individu of groep']=='mijzelf']
    data_org = data_vraag[data_vraag['individu of groep']!='mijzelf']

    vraagTokens = data_vraag['SentToks'].tolist()

    dic = gensim.corpora.Dictionary(vraagTokens)
    corpus = [dic.doc2bow(text) for text in vraagTokens]
    return (dic, corpus, data_ppl)


def clean_plaintext_corpus(text):
	stopwords = nltk.corpus.stopwords.words('dutch')
	stopwords += nltk.corpus.stopwords.words('english')
	stopwords += ['','.',',','?','(',')',',',':',"'",u'``',u"''",';','-','!','%','&','...','=','we','wij']
	stopwords += [u'\u2019',u'\u2018',u'\u2013',u'\u2022',u'\u2014',u'\uf02d',u'\u20ac',u'\u2026']

	keepTags = [
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
	    u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ' # Verb forms
	]
	return filter_nlp_tags(text, keepTags, stopwords)

def filter_nlp_tags(text, nlp_tags, exclude_words = []):
    words = []
    for s in split(parse(sentText)):
        for word, tag in s.tagged:
            if tag in nlp_tags:
                words.append(word.lower())
    words = [ w for w in words if w not in exclude_words ]
    return words
