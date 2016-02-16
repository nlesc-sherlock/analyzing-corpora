rm(list=ls())
# install.packages("LDAvis")
require(Matrix)
library(topicmodels) 
library(LDAvis)

# load example data and run LDA
setwd("~/sherlock/topic group/github/analyzing-corpora/")
pathapp = "scripts/shinyapp/"
if (file.exists(paste0(pathapp,"M.RData")) == FALSE) {
  fname =  "data/VraagTextCorpus.mm"# "~/sherlock2/data/VraagTextCorpus.mm"
  M = readMM(file=fname)
  S = LDA(x=M,k=5)
  pos = posterior(S)
  save(M,S,pos,file=paste0(pathapp,"M.RData"))
} else {
  print("load previously saved data")
  load(paste0(pathapp,"M.RData"))
}

# reshape data to work with visualisation
theta  = pos$topics # documenst x topics  matrix
phi = pos$term# topics x terms matrix
a = summary(M)
doc.length <- sapply(split(a,a$i),function(x) nrow(x))  # number of words per file
term.table = sort(table(a$j),decreasing=TRUE) # table of terms
vocab = names(term.table) # term table
term.frequency <- as.integer(term.table) 

# create the JSON object to feed the visualization:
json <- createJSON(phi = phi, 
                   theta = theta, 
                   doc.length = doc.length, 
                   vocab = vocab, 
                   term.frequency = term.frequency)


serVis(json, out.dir = '~/sherlock/topic group/repository/myvis', open.browser = TRUE)