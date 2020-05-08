rm(list = ls())

# Read Data
data <- read.csv("C:/Users/robna/Desktop/bankruptcy_Train.csv")
str(data)
data$class <- as.factor(data$class)
table(data$class)

# Data Partition
set.seed(123)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train <- data[ind==1,]
test <- data[ind==2,]

# Random Forest
library(randomForest)
set.seed(222)
rf <- randomForest(class~., data=train,
                   ntree = 500,
                   mtry = 8,
                   importance = TRUE,
                   proximity = TRUE)
print(rf)
attributes(rf)

# Prediction & Confusion Matrix - train data
library(caret)
p1 <- predict(rf, train)
confusionMatrix(p1, train$class)

# # Prediction & Confusion Matrix - test data
p2 <- predict(rf, test)
confusionMatrix(p2, test$class)

# Error rate of Random Forest
plot(rf)

# Tune mtry
t <- tuneRF(train[,-65], train[,65],
       stepFactor = 0.5,
       plot = TRUE,
       ntreeTry = 500,
       trace = TRUE,
       improve = 0.05)

# No. of nodes for the trees
hist(treesize(rf),
     main = "No. of Nodes for the Trees",
     col = "green")

# Variable Importance
varImpPlot(rf,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf)
varUsed(rf)


# Extract Single Tree
getTree(rf, 1, labelVar = TRUE)

# Multi-dimensional Scaling Plot of Proximity Matrix
MDSplot(rf, train$class)
