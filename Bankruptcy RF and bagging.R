
# Bagging and Random Forest Example Bankruptcy

rm(list = ls())

library (randomForest)
Bankruptcy = read.csv("C:/Users/robna/Desktop/bankruptcy_Train.csv")

# Training set
set.seed(1)    # Produce the same sample
train = sample(1:nrow(Bankruptcy), nrow(Bankruptcy)*.8)

# Convert Dependent Variable to Factor
Bankruptcy$class = as.factor(Bankruptcy$class)

# Test set
Bankruptcy.test = Bankruptcy[-train,"class"]

# Bagging
bag.Bankruptcy = randomForest(class~., data = Bankruptcy, subset = train, mtry = 64, ntree = 500)
bag.Bankruptcy

# Predict the test set
yhat.bag = predict(bag.Bankruptcy, newdata = Bankruptcy[-train,])

yhat.bag[yhat.bag<=0.5] <- 0
yhat.bag[yhat.bagt>0.5] <- 1

# Calculate the prediction accuracy
T_bag <- table(Bankruptcy.test, yhat.bag)
T_bag   # Display the contingency table
Accuracy_bag <- 1 - mean(Bankruptcy.test != yhat.bag)
Accuracy_bag   # Display the prediction accuracy









# Random Forest
rf.Bankruptcy = randomForest(class~., data = Bankruptcy, subset = train, mtry = 8, ntree = 500)
rf.Bankruptcy

# Predict the test set
yhat.rf = predict(rf.Bankruptcy, newdata = Bankruptcy[-train,])
mean((yhat.rf-Bankruptcy.test)^2)   # MSE

