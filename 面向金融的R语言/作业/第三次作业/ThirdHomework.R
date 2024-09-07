#setwd("D:\\Study\\研一\\面向金融的R语言\\作业\\第三次作业")

#install.packages("caret")
#install.packages("kernlab")
#install.packages("ellipse")
#install.packages("randomForest")

# 导入必要的库
library(caret)
library(kernlab)
library(randomForest)
library(ellipse)


# 读取数据集
data <- read.csv("homework3_data.csv")

data$Weekend <- as.factor(data$Weekend)
data$Revenue <- as.factor(data$Revenue)

# 将数据集划分为训练集和测试集
set.seed(123)  # 设置随机种子以确保结果可复现
trainIndex <- createDataPartition(data$Revenue, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# 使用11折交叉验证
ctrl <- trainControl(method = "cv", number = 5)


# KNN
model_knn <- train(Revenue ~ ., data = trainData, method = "knn", trControl = ctrl)
knn_results <- predict(model_knn, testData)

# LDA
model_lda <- train(Revenue ~ ., data = trainData, method = "lda", trControl = ctrl)
lda_results <- predict(model_lda, testData)

# GLM
model_glm <- train(Revenue ~ ., data = trainData, method = "glm", trControl = ctrl)
glm_results <- predict(model_glm, testData)

# CART
model_tree <- train(Revenue ~ ., data = trainData, method = "rpart", trControl = ctrl)
tree_results <- predict(model_tree, testData)

# Random Forest
model_rf <- train(Revenue ~ ., data = trainData, method = "rf", trControl = ctrl)
rf_results <- predict(model_rf, testData)

# SVM
model_svm <- train(Revenue ~ ., data = trainData, method = "svmRadial", trControl = ctrl)
svm_results <- predict(model_svm, testData)


# 比较不同模型的性能
#confusionMatrix(knn_results, testData$Revenue)
#confusionMatrix(lda_results, testData$Revenue)
#confusionMatrix(glm_results, testData$Revenue)
#confusionMatrix(tree_results, testData$Revenue)
#confusionMatrix(rf_results, testData$Revenue)
#confusionMatrix(svm_results, testData$Revenue)
