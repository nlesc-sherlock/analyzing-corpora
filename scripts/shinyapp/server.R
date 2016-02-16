library(shiny)
require(wordcloud)
#------------------------
# get data
pathM = "~/sherlock/topic group/"
# pathapp = "~/sherlock/topic group/github/analying-corpora/repository/scripts/shinyapp/"

load(paste0(pathM,"M.RData"))
dic = read.csv(paste0(pathM,"filtered_0.1_5_1000000.dic.txt"),sep="\t",header=FALSE)
names(dic) =c("number","word","count")


theta  = pos$topics # documenst x topics  matrix
phi = pos$term# topics x terms matrix
a = summary(M)
a = merge(a,dic,by.y="number",by.x="j")
doc.length <- sapply(split(a,a$i),function(x) nrow(x))  # number of words per file
docid = unique(a$i)
# term.table = sort(table(a$j),decreasing=TRUE) # table of terms
term.table = table(a$word)

vocab = names(term.table) # term table



term.frequency <- as.integer(term.table)
theta.c <- apply(theta, 1, which.max)
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
      sel = which(x >= input$range[1] & x <= input$range[2] & theta.c == 1)
      A = sort(table(a$j[which(a$i %in% docid[sel] == TRUE)]),decreasing=TRUE) # table of terms
      if (length(A) > 10) { # only look at most frequent 50 words
        threshold = sort(A,decreasing=TRUE)[10]
        B = which(A >= threshold)
      } else {
        B = length(A)
      }
      plot.new()
      text(x=0.5,y=0.5,labels=paste0("Topic ",i),font=2,cex=3)
      wordcloud(words=names(A[B]),freq=A[B],rot.per=0.01)    
    }
  })
})
