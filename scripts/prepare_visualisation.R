rm(list=ls())

setwd("~/sherlock/topic group/github/analyzing-corpora/") #set working directory
pathM = "scripts/shinyapp/" # directory of shinyapp
datadir = "data/" #directory of dictionary

# GET WORD DICTIONARY
dic = read.csv(file=paste0(datadir,"enron_dic.csv"),stringsAsFactors=FALSE,sep ="\t",header=FALSE) 
names(dic) =c("id","word")

# GET WORDS X TOPICS
woto = read.csv(paste0(datadir,"enron_lda_15.csv"),header=FALSE) 

# GET DOCUMENTS x TOPICS
doto = read.csv(file=paste0(datadir,"enron_document_topics_15.csv"),sep ="\t",header=FALSE) 

# GET DOCUMENT DICTIONARY (id -> filename)
# ... not available yet

# derive summaries for wordxtopic
RS1 = rowSums(woto)
wotonorm = as.data.frame(woto / RS1) # normalize probabilities per word
winningwoto = max.col(wotonorm)     # <--- most popular topic per word
Nwordsto = table(winningwoto)   # <---  number of (winning) words per topic
wotoorder = matrix(0,nrow(wotonorm),ncol(wotonorm))
wotoorder_words = matrix("",nrow(wotonorm),ncol(wotonorm))
wotonorm$id = 1:nrow(wotonorm)
for (i in 1:(ncol(wotonorm)-1)) {
  wotoorder[,i] = wotonorm$id[order(-wotonorm[,i])]   # <--- ranking of wordsindeces by their distance to a topic
  wotoorder_words[,i] = dic[wotoorder[,i],2]          # <--- ranking of words by their distance to a topic
}
# derive summaries for documentxtopic
RS2 = rowSums(doto)
winningtodo = max.col(doto)     # <--- most popular topic per document
Ndocsto = table(winningtodo)    # <--- number of (winning) documents per topic
dotonorm = as.data.frame(doto / RS2) # normalize probabilities per word
dotoorder = matrix(0,nrow(dotonorm),ncol(dotonorm))
dotonorm$id = 1:nrow(dotonorm)
for (i in 1:(ncol(dotonorm)-1)) {
  dotoorder[,i] = dotonorm$id[order(-dotonorm[,i])]   # <--- ranking of documents by topics
}