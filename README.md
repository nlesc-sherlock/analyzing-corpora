# Analyzing Corpora
For project Sherlock, our team aims to use NLP tools to analyse large collections of documents. The original description of the team's goals are on [Sherlock's repo](https://github.com/NLeSC/Sherlock/blob/master/topics/analyzing_document_collections/analyzing_large_document_collections.md).

## Tools
We will be working using Python. We might be using the following libraries:
 - gensim==0.12.0
 - nltk==3.0.5
 - wordcloud==1.1.3
 - scikit-learn==0.17.dev0

These can be installed with pip (if needed in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/)):

```shell
make install
```

## Dataset

A good example data set is the Enron email archive. This data set can be downloaded from [here](https://www.cs.cmu.edu/~./enron/).

## How to run

First an email folder is preprocessed to remove MIME headers and keep only the message contents.

```shell
python scripts/cleanHeaders.py email_folder cleaned_email_folder
```

Then the cleaned email will be parsed to a single corpus
```shell
python scripts/parse_email.py -d dictiononary.dic -m metadata.h5 cleaned_email_folder corpus.pkl
```
If the `-m` flag is given, metadata is stored in HDF5 format. In that case, HDF5 should be installed on the system. The dictionary contains all words that are occur in the corpus.

Words that are too frequent (occurs in more than `-a` fraction of the documents) or not sufficiently frequent (occurs in less than `-b` documents) may be removed in order not to pollute the topic space. After that, only the `-k` most frequent words are kept.
```shell
python scripts/filter_extremes.py -a 0.1 -b 5 -k 100000 dictionary.dic filtered_dictionary.dic
```
After this operation, use filtered_dictionary.dic only.

Once we have a final dictionary, we can generate a sparse matrix of documents times words
```shell
python scripts/corpus_to_sparse.py -d filtered_dictionary.dic corpus.pkl corpus_matrix.npz
```

This can then be used to run the LDA on, where the LDA object will be written to `output_folder`:
```shell
mkdir output_folder
python scripts/create_lda.py -s corpus_matrix.npz -d filtered_dictionary.dic -m metadata.h5 --topics 15 output_folder
```

## Further reading

This (fairly recent) [paper](http://idl.cs.washington.edu/papers/topic-check/) talks about the issues with topic model stability -- would be interesting to read and see what we can learn from them.
