# Analyzing Corpora
For project Sherlock, our team aims to use NLP tools to analyse large collections of documents. The original description of the team's goals are on [Sherlock's repo](https://github.com/NLeSC/Sherlock/blob/master/topics/analyzing_document_collections/analyzing_large_document_collections.md).

## Tools
We will be working using Python. We might be using the following libraries:
 - gensim==0.12.0
 - nltk==3.0.5
 - wordcloud==1.1.3
 - scikit-learn==0.17.dev0

These can be installed with pip (if needed in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/)):

    pip install -r requirements.txt

## Dataset
We will be working with the Enron data set. The original data set can be downloaded from [here](https://www.cs.cmu.edu/~./enron/). However, for simplicity, there is a pre-processed version of the data set on [Sherlock's OneDrive](https://nlesc.sharepoint.com/sites/sherlock/_layouts/15/Group.aspx?GroupId=6aad52c4-7dfc-4076-9772-4f9c9180bde2&AppId=Files&id=%2Fsites%2Fsherlock%2FShared%20Documents%2Fdatasets%2Fenron-plaintext).


## Further reading

This (fairly recent) [paper](http://idl.cs.washington.edu/papers/topic-check/) talks about the issues with topic model stability -- would be interesting to read and see what we can learn from them.

Other implementation of LDA is available in [spark](https://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda) which should be more parallel than gensim or scikit -- worth trying?
