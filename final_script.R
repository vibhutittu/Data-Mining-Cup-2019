trainData <- read.csv("train.csv",sep="|",stringsAsFactors = FALSE)
testData <- read.csv("test.csv", sep="|",stringsAsFactors = FALSE)

set.seed(100)
trainData$fraud <- as.factor(trainData$fraud)
levels(trainData$fraud) <- c("trusty","fraud")


#######################################
##Ensemble model
#######################################
library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)


algorithmList <- c('rf','xgbDART', 'svmRadial')
set.seed(100)
models <- caretList(fraud~., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)
save(models,file="models_final.RData")

##################################################################
####Combine the predictions of models to form final prediction
##################################################################
set.seed(101)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
save(stack.glm,file="combined_predictions.RData")
print(stack.glm)


# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=testData)
head(stack_predicteds)
write.csv(stack_predicteds,file="predictions.csv")
save.image("script.RData")

levels(stack_predicteds) <- c("1","0")
testData$fraud <- stack_predicteds
write.table(testData,file="testData_predictions.csv",row.names=FALSE,sep="|")
