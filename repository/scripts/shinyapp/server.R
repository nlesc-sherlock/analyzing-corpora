library(shiny)
library(wordcloud)
#------------------------
# get data
load("M.RData") #~/sherlock2/repository/scripts/shinyapp/M.RData"
# dic = read.csv("~/sherlock2/filtered_0.1_5_1000000.dic.txt",sep="\t",header=FALSE)
# names(dic) =c("number","word","count")
# theta  = pos$topics # documenst x topics
# phi = pos$term # topics x terms
# theta.c <- apply(theta, 1, which.max) # define topic from highest probability
# a = summary(M)
# a = merge(a,dic,by.y="number",by.x="j")
# doc.length <- sapply(split(a,a$i),function(x) nrow(x))  # number of words per file across 11691 files
# save(dic,theta,phi,theta.c,a,doc.length,M,S,pos,file="M.RData")
docid = unique(a$i)
term.table = sort(table(a$j),decreasing=TRUE) # frequency table of words
vocab = names(term.table) # term table
dict = unique(a[,c('word','j')])
dict$word = as.character(dict$word)
term.frequency <- as.integer(term.table)


#------------------------
shinyServer(function(input, output) {
  x = doc.length
  output$topicControls <- renderUI({
    sel = which(x >= input$range[1] & x <= input$range[2])
    checkboxGroupInput("topics", "Choose Topics (max 3)", names(table(theta.c[sel])))
  })
  output$rangeControls <- renderUI({
    sliderInput("range", "Range in word count per document:",
                min = 0, max = max(doc.length), value = range(doc.length))
  })
  output$distPlot <- renderPlot({
    bins <- seq(input$range[1], input$range[2], length.out = 31) #input$bins
    x = x[which(x >= input$range[1] & x <= input$range[2])]
    hist(x, xlim=input$range,breaks = bins, col = 'lightblue', border = 'white',
         xlab="word count",main="word count distribution",cex.lab=1.5)
  })
  output$topPlot <- renderPlot({
    sel = which(x >= input$range[1] & x <= input$range[2])
    xx = table(theta.c[sel])
    barplot(xx,xlim=c(0,length(xx)+1),ylab="Number of documents",
            xlab="Topic",col = 'lightblue', border = 'white',cex.lab=1.5)    
  })
  output$wordPlot <- renderPlot({
    if (length(input$topics) > 3) {
      itopics = input$topics[1:3]
    } else {
      itopics = input$topics
    }
    par(mar=rep(0,4),mfrow=c(2,3))
    mat = 1:6
    dim(mat) = c(2,3)
    layout(mat=mat, heights=c(1, 4))
    for (i in itopics) {
      sel = which(x >= input$range[1] & x <= input$range[2] & theta.c == i)
      A = sort(table(a$j[which(a$i %in% docid[sel] == TRUE)]),decreasing=TRUE) # frequency of words
      if (length(A) > 10) { # only look at most frequent 50 words
        threshold = sort(A,decreasing=TRUE)[10]
        B = which(A >= threshold)
      } else {
        B = length(A)
      }
      plot.new()
      text(x=0.5,y=0.5,labels=paste0("Topic ",i),font=2,cex=3)
      wordlist = as.numeric(names(A[B]))
      wordlist.names = rep("",length(wordlist))
      for (h in 1:length(wordlist)) {
        wordlist.names[h] = as.character(dict$word)[which(dict$j == wordlist[h])][1]
      }
      wordcloud(words=wordlist.names,freq=A[B],rot.per=0.01,random.order=FALSE)    #names(A[B])
    }
  })
})
