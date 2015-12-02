rm(list=ls())
graphics.off()

# based on:
# http://stackoverflow.com/questions/16396090/r-topic-modeling-lda-model-labeling-function
# http://stackoverflow.com/questions/16115102/predicting-lda-topics-for-new-data
# http://www.rtexttools.com/blog/getting-started-with-latent-dirichlet-allocation-using-rtexttools-topicmodels
# not implemented, but possibly interesting to look at: https://ropensci.org/blog/2014/04/16/topic-modeling-in-R/

library(topicmodels)
require(Matrix)

print("get data")
fname =  "D:/sherlock_topic/VraagTextCorpus.mm"
M = readMM(file=fname)

print("train model on subsample")
ind = 1:nrow(M)
sam = sample(ind,size=1000,replace=FALSE)
res = which(ind %in% sam == FALSE)
S = LDA(x=M[sam,],method="VEM",k=3,verbose=1,VEMcontrol=list(seed=123,verbose=1,iter=10,tol=10^-3))
print("classify entire dataset")
lda_inf = posterior(S, M[res,])