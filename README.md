# Analyzing Corpora
For project Sherlock, our team aims to use NLP tools to analyse large collections of documents. The original description of the team's goals are on [Sherlock's repo](https://github.com/NLeSC/Sherlock/blob/master/topics/analyzing_document_collections/analyzing_large_document_collections.md).

The following sections describes the process for going from a bunch of plain text documents (emails in this case) to a nice visualization of the topics in these documents.

## Tools
We are working in a mixture of Python, Scala, Spark, R, and other tools. Setup instructions for each of these tools is described here.

### Python setup
Python dependencies can be with pip (in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if desired):

```shell
make install
```

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

### Spark setup
How do you get spark running locally? Is it necessary? Or is this optional (because we are using Spark on the cluster)

### Forqlift
[Forqlift](http://www.exmachinatech.net/projects/forqlift/) is a tool for converting plain text files to sequence files. HDFS (and thus spark) does not work well with lots of small files, so sequence files are used instead.

To install forqlift, simply [download](http://www.exmachinatech.net/projects/forqlift/download/) the binaries and extract them. Add `$FORQLIFT/bin` to your `PATH` and you are ready to run `forqlift`.

## Dataset

A good example data set is the Enron email archive. This data set can be downloaded from [here](https://www.cs.cmu.edu/~./enron/).

## Step 1 - The original data

The initial enron email data set can be found [here](https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz). This compressed file contains plain text emails. Use `forqlift` to create a sequence file:

    forqlift fromarchive enron_mail_20150507.tgz --file enron_mail.seq --compress bzip2 --data-type text

## Step 2 - Preprocessing
Data preparation: build dictionary, convert documents go bag-of-words (correct?). Which script do we need to run this (EmailParser.scala?)?

   ** Inputs: ** `enron_mail.seq`  
   ** Outputs: ** `? Dictionary, metadata, bags of words ?`

## Step 3 - Run LDA: document how this is run.
   - This step could be run multiple times (for different number of topics).

   ** Inputs: ** `? Dictionary, metadata, bags of words ?`  
   ** Outputs: ** `? Topic model ?`

## Step 4 - Visualization

### Step 4.a - Run clustering / visualization (IPython notebook)
### Step 4.b - Run R-shiny visualization


## Further reading

This [paper](http://idl.cs.washington.edu/papers/topic-check/) talks about the issues with topic model stability -- would be interesting to read and see what we can learn from them.
