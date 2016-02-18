rm(list=ls())
setwd("~/sherlock/topic group/github/analyzing-corpora/") #set working directory
pathM = "scripts/shinyapp/" # directory of shinyapp
datadir = "data/" #directory of dictionary

source("scripts/prepare_visualisation.R") # directory of shinyapp

# GET WORD DICTIONARY
dic = read.csv(file=paste0(datadir,"enron_dic.csv"),stringsAsFactors=FALSE,sep ="\t",header=FALSE) 
names(dic) =c("id","word")
# GET WORDS X TOPICS
woto = read.csv(paste0(datadir,"enron_lda_15.csv"),header=FALSE) 
# GET DOCUMENTS x TOPICS
doto = read.csv(file=paste0(datadir,"enron_document_topics_15.csv"),sep ="\t",header=FALSE) 
# GET DOCUMENT DICTIONARY (id -> filename)
# ... not available yet

out = prepare_visualisation(dic,woto,doto)