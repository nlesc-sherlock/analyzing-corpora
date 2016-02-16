# Analyzing Corpora
For project Sherlock, our team aims to use NLP tools to analyse large collections of documents. The original description of the team's goals are on [Sherlock's repo](https://github.com/NLeSC/Sherlock/blob/master/topics/analyzing_document_collections/analyzing_large_document_collections.md).

The following sections describes the process for going from a bunch of plain text documents (emails in this case) to a nice visualization of the topics in these documents.

## Tools
We are working in a mixture of Python, Scala, Spark, R, and other tools. Setup instructions for each of these tools is described here.

### Python setup
Python dependencies can be with pip (in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if desired):

    pip install -r requirements.txt

### Spark setup
How do you get spark running locally? Is it necessary? Or is this optional (because we are using Spark on the cluster)

### Forqlift
[Forqlift](http://www.exmachinatech.net/projects/forqlift/) is a tool for converting plain text files to sequence files. HDFS (and thus spark) does not work well with lots of small files, so sequence files are used instead.

To install forqlift, simply [download](http://www.exmachinatech.net/projects/forqlift/download/) the binaries and extract them. Add `$FORQLIFT/bin` to your `PATH` and you are ready to run `forqlift`.

## Dataset
We will be working with the Enron data set. The original data set can be downloaded from [here](https://www.cs.cmu.edu/~./enron/). For simplicity, there is a pre-processed (email headers removed) version of the data set on [Sherlock's OneDrive](https://nlesc.sharepoint.com/sites/sherlock/_layouts/15/Group.aspx?GroupId=6aad52c4-7dfc-4076-9772-4f9c9180bde2&AppId=Files&id=%2Fsites%2Fsherlock%2FShared%20Documents%2Fdatasets%2Fenron-plaintext).

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
