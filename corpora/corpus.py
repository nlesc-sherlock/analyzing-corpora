import gensim
import pickle
import pandas as pd
import nltk
from pattern.nl import parse, split
import sys
import os
from time import time

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
	return filter_nlp_tags(text, keepTags, exclude_words=stopwords)


def filter_nlp_tags(text, nlp_tags, exclude_words = []):
    words = []
    for s in split(parse(text)):
        for word, tag in s.tagged:
            if tag in nlp_tags:
                words.append(word.lower())
    words = [ w for w in words if w not in exclude_words ]
    return words


def load_enron_corpus(directory):
	total_size = 0
	for root, dirs, files in os.walk(directory):
		total_size += len(files)

	print("reading {0} files".format(total_size))

	start_time = time()
	documents = []
	for user in os.listdir(directory):
		print("User: {0}".format(user))
		for root, dirs, files in os.walk(os.path.join(directory, user)):
			mailbox = os.path.basename(root)
			if len(files) > 0:
				print("Mailbox: {0} ({1:.0%}, {2:.2f} seconds)".format(mailbox, len(documents)/float(total_size), time() - start_time))
			for email in files:
				doc = {'user': user, 'mailbox': mailbox, 'directory': root}
				with open(os.path.join(root, email)) as f:
					doc['tokens'] = clean_plaintext_corpus(f.read())
				documents.append(doc)

	return documents


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: corpus.py enron_directory enron_pickle_file.pkl")
		sys.exit(1)

	docs = load_enron_corpus(sys.argv[1])
	print("Emails: {0}".format(len(docs)))
	with open(sys.argv[2], 'wb') as f:
		pickle.dump(docs, f)
