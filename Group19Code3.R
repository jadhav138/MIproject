library(shiny)
library(randomForest)
library(caret)
library(ggplot2)
library(sparklyr)

# Initialize Spark connection
sc <- spark_connect(master = "local")

# Read data
filePath <- "genetic_markers_dataset.csv"
df <- spark_read_csv(sc, "df", filePath, header = TRUE, infer_schema = TRUE)

# Function to manually scale a column
scale_column <- function(df, column_name) {
  mean_val <- df %>% summarize(mean = mean(!!sym(column_name))) %>% collect() %>% .[["mean"]]
  sd_val <- df %>% summarize(sd = sd(!!sym(column_name))) %>% collect() %>% .[["sd"]]
  
  scaled_column_name <- paste0(column_name, "_scaled")
  df %>% mutate(!!scaled_column_name := (!!sym(column_name) - mean_val) / sd_val)
}

# UI definition
ui <- fluidPage(
  titlePanel("Disease Likelihood Prediction with Random Forest"),
  sidebarLayout(
    sidebarPanel(
      fileInput('datafile', 'Upload CSV File', accept = c(".csv")),
      actionButton("train", "Train Model")
    ),
    mainPanel(
      DT::dataTableOutput('tableData')
    )
  )
)

# Server logic
server <- function(input, output, session) {
  observeEvent(input$goButton, {
    req(input$file1)
    
    trainedModel <- eventReactive(input$train, {
      data <- dataInput()
      if (is.null(data)) return(NULL)
      
      # Data scaling
      scaled_data <- scale_column(data, "feature_column_name")
      
      # Splitting data into features and target variable
      X <- subset(scaled_data, select = -Disease_Likelihood)
      y <- scaled_data$Disease_Likelihood
      
      # Model training logic
      set.seed(42)
      trainIndex <- caret::createDataPartition(y, p = 0.7, list = FALSE)
      X_train <- X[trainIndex, ]
      y_train <- y[trainIndex]
      X_val <- X[-trainIndex, ]
      y_val <- y[-trainIndex]
      
      model <- randomForest(x = X_train, y = y_train)
      saveRDS(model, "model.rds")
      list(model = model, X_val = X_val, y_val = y_val)
    })
    
    output$tableData <- DT::renderDataTable({
      data <- dataInput()
      if (is.null(data)) return()
      data
    })
  })
}

# Run the application 
shinyApp(ui, server)
