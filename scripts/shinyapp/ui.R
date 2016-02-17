library(shiny)

# runApp("~/sherlock2/repository/scripts/shinyapp",display.mode = "showcase")

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Hello Sherlock!"),
  
  # Sidebar with a slider input for the number of bins
  sidebarLayout(
    sidebarPanel(
      # uiOutput("rangeControls"),
      uiOutput("topicControls")
    ),
    # Show a plot of the generated distribution
    mainPanel(
      # plotOutput("distPlot",height=250),
      # plotOutput("topPlot",height=250),
      plotOutput("wordPlot",height=500)
    )
  ),
  fluidRow(
    column(10, wellPanel(
      "Example message",
      verbatimTextOutput("fileReaderText")
    ))
  )
))
