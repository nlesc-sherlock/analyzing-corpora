import gensim
import pickle

def load_vraagtekst_corpus(documents_filename):
    with open(documents_filename, 'r') as f:
        data_vraag = pickle.load(f)
    data_ppl = data_vraag[data_vraag['individu of groep']=='mijzelf']
    data_org = data_vraag[data_vraag['individu of groep']!='mijzelf']

    vraagTokens = data_vraag['SentToks'].tolist()

    dic = gensim.corpora.Dictionary(vraagTokens)
    corpus = [dic.doc2bow(text) for text in vraagTokens]
    return (dic, corpus, data_ppl)
