File enron_vocab.gensim.dic is a gensim vocabulary dictionary for the enron 
email data set. This file contains word counts of all existing words. It can 
be loaded as follows:

  import gensim

  dic = gensim.corpora.Dictionary.load('enron_vocab.gensim.dic')

And used to convert a document to a bag-of-words:

  # Notice that texcatlipocatl does not appear in the corpus
  email = 'hello all this is a sample email from texcatlipocatl'.split()
  emailIds = dic.doc2bow(email)
  for id,count in emailIds:
    print 'Word  %-10s (id=%6d) appears %6d times in corpus'%('"'+dic[id]+'"',id,dic.dfs[id])

The dictionary can be filtered to remove words we are willing to ignore:

  # Let's remove stop words
  stopwords = 'all this is a from '.split()
  stopwordIds = [ id for id,count in dic.doc2bow(stopwords) ]
  dic.filter_tokens(bad_ids=stopwordIds)

  email = 'hello all this is a sample email from texcatlipocatl'.split()
  emailIds = dic.doc2bow(email)
  for id,count in emailIds:
    print 'Word  %-10s (id=%6d) appears %6d times in corpus'%('"'+dic[id]+'"',id,dic.dfs[id])


