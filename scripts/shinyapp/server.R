library(shiny)
require(wordcloud)



# set directories
setwd("~/sherlock/topic group/github/analyzing-corpora/") #set working directory
pathM = "scripts/shinyapp/" # directory of shinyapp
datadir = "data/" #directory of dictionary

print("load data")
# unique words x topics [175884 x 15]
lda = read.csv(paste0(datadir,"enron_lda_15.csv"),header=FALSE) 
phi = t(lda)

# unique words id and word [175884 x 2]
dic = read.csv(file=paste0(datadir,"enron_dic.csv"),sep ="\t",header=FALSE) 
# dic = read.csv(file=paste0(datadir,"filtered_0.1_5_1000000.dic.txt"),sep ="\t",header=FALSE) 
# names(dic) =c("id","word","count")
names(dic) =c("id","word")

#  i (document), j (word), x (count)
# a = read.csv(file=paste0(datadir,"sparse_matrix.csv"),sep ="\t",header=FALSE) 
# names(a) = c("i","j","x") 
# # theta [documents x topics] 
# theta = read.csv(file=paste0(datadir,"enron_document_topics_15.csv"),sep ="\t",header=FALSE) 
# # udoc = unique(a$doc)
# # utop = nrow(phi)
# print("prepare data for shiny")
# #-----------------------------
# # create additional variables
# a = merge(a,dic,by.y="number",by.x="j") # merge in labels
# doc.length <- sapply(split(a,a$i),function(x) nrow(x))  # number of words per file
# docid = unique(a$i) # unique file ids
# # term.table = sort(table(a$j),decreasing=TRUE) # table of terms
# term.table = table(a$word) # how often does every word occur in all the documents
# vocab = names(term.table) # unique words
# term.frequency <- as.integer(term.table) # frequency of unique words
# theta.c <- apply(theta, 1, which.max) # most popular topic per document

pop.pr = pop.words = matrix(0,nrow(phi),20)
for (i in 1:nrow(phi)) {
  yy = sort(phi[i,],decreasing=TRUE)[20]
  popwords = which(phi[i,] >= yy)
  ppwd = as.character(dic$word[which(dic$id %in% popwords == TRUE)])
  pop.words[i,] = ppwd
  pop.pr[i,] = phi[i,popwords]
}

topicnames = 1:15 #paste0("T",1:20)

#------------------------
shinyServer(function(input, output) {
  # x = doc.length
  output$topicControls <- renderUI({
    # sel = which(doc.length >= input$range[1] & doc.length <= input$range[2])
    # checkboxGroupInput("topics", "Choose Topics (max 3)", names(table(theta.c[sel])))
    checkboxGroupInput("topics", "Choose Topics (max 3)", topicnames)
  })
#   output$rangeControls <- renderUI({
#     sliderInput("range", "Range in word count per document:",
#                 min = 0, max = max(doc.length), value = range(doc.length))
#   })
#   output$distPlot <- renderPlot({ #
#     bins <- seq(input$range[1], input$range[2], length.out = 31) #input$bins
#     doc.length = doc.length[which(doc.length >= input$range[1] & doc.length <= input$range[2])]
#     hist(doc.length, xlim=input$range,breaks = bins, col = 'lightblue', border = 'white',
#          xlab="word count",main="word count distribution",cex.lab=1.5)
#   })
#   output$topPlot <- renderPlot({ # possibly not relevant
#     sel = which(doc.length >= input$range[1] & doc.length <= input$range[2])
#     xx = table(theta.c[sel])
#     barplot(xx,xlim=c(0,length(xx)+1),ylab="Number of documents per topic",
#             xlab="Topic",col = 'lightblue', border = 'white',cex.lab=1.5)    
#   })
  output$wordPlot <- renderPlot({ # word cloud... replace by cloud based on phi
    if (length(input$topics) > 3) {
      itopics = as.numeric(input$topics[1:3])
    } else {
      itopics = as.numeric(input$topics)
    }
    
    print(input$topics)
    
    par(mar=rep(0,4),mfrow=c(2,3))
    mat = 1:6
    dim(mat) = c(2,3)
    layout(mat=mat, heights=c(1, 4))
    for (i in itopics) {
      # sel = which(doc.length >= input$range[1] & doc.length <= input$range[2] & theta.c == i) #derive index of documents with word count within range and belonging to topic i
#       A = sort(table(as.character(a$word[which(a$i %in% docid[sel] == TRUE)])),decreasing=TRUE) # derive words belong to those documents and sort them
#       if (length(A) > 6) { # only look at most frequent 50 words
#         threshold = sort(A,decreasing=TRUE)[6]
#         B = which(A >= threshold)
#       } else {
#         B = length(A)
#       }
      plot.new()
      text(x=0.5,y=0.5,labels=paste0("Topic ",i),font=2,cex=3)
      # wordcloud(words=names(A[B]),freq=A[B],rot.per=0.01)
    
      freqvals = as.numeric(pop.pr[i,]) #/min(as.numeric(pop.pr[i,]))
      freqvals  = freqvals / max(freqvals)
      wordcloud(words=as.character(pop.words[i,]),freq=freqvals,rot.per=0.01)    
    }
# }
  })
})
