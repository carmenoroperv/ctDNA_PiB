library(tidyverse)
library(caret)
library(pROC)
library(xgboost)
library(doParallel)
library(foreach)
library(doRNG)
library(multiROC)
library(dummies)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
library(splitTools)

system("R --max-ppsize=100000 --save")
options(expressions = 500000)

data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)

fold <- readRDS(snakemake@input[["input_fold"]])
cv_rep <- as.numeric(snakemake@params[["cv_rep"]])

data <- data %>% filter(sample_type != "Healthy")
message(unique(data$sample_type))
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data <- data %>% droplevels(c("Duodenal_Cancer", "Healthy"))

message(data %>% group_by(sample_type) %>% summarize(n = n()))
message(levels(data$sample_type))

observed  <- data$sample_type

data$sample_type <- as.factor(data$sample_type)
print("Sample_type levels")
levels(data$sample_type)

print("Head of fold (train rows)")
print(fold[0:6])

message(paste("CV repetition number: ", cv_rep, sep = ""))
set.seed(cv_rep)

testdata  <- data[-fold,]
testdata <- testdata %>% dplyr::select(-sample_type)
traindata <- data[fold,]
trainlabels <- traindata$sample_type
traindata <- traindata %>% dplyr::select(-sample_type)

rows <- data %>% mutate(row_name = row_number()) %>% dplyr::select(row_name)
test_rows <- rows[-fold,]

print("Head of test fold (test_rows)")
print(head(test_rows))

predicted <- tibble(row_predicted = test_rows,
             Bile_Duct_Cancer = rep(NA, nrow(data) - length(fold)),
             Breast_Cancer = rep(NA, nrow(data) - length(fold)),
             Colorectal_Cancer = rep(NA, nrow(data) - length(fold)),
             Gastric_cancer = rep(NA, nrow(data) - length(fold)),
             Lung_Cancer = rep(NA, nrow(data) - length(fold)),
             Ovarian_Cancer = rep(NA, nrow(data) - length(fold)),
             Pancreatic_Cancer = rep(NA, nrow(data) - length(fold)))


################# Nested cross validation #######################
set.seed(0)
seeds <- vector(mode = "list", length = 11)
for(i in 1:10) seeds[[i]]<- sample.int(n=1000, 18)
#for the last model
seeds[[11]]<-sample.int(1000, 1)

trControl_gbm <- trainControl(method = "repeatedcv", 
                              seeds = seeds,
                              number = 10, 
                              repeats = 1, 
                              classProbs = TRUE, 
                              allowParallel=TRUE)

#gbmGrid <- expand.grid(interaction.depth = c(1, 2, 3),
#                       n.trees = seq(200, 800, 200),
#                       shrinkage = c(0.1, 0.2, 0.01),
#                       n.minobsinnode = c(10))


fit1 <- train(x = traindata, 
              y = trainlabels, 
              method = "gbm",
              tuneLength = 5,
              trControl = trControl_gbm, 
              verbose=F)

message("besttune n.trees")
message(fit1$bestTune$n.trees)
message("besttune interaction.depth")
message(fit1$bestTune$interaction.depth)
message("besttune shrinkage")
message(fit1$bestTune$shrinkage)
message("besttune n.minobsinnode")
message(fit1$bestTune$n.minobsinnode)
#################################################################

fitControl <- trainControl(classProbs = TRUE)
fit2 <- train(x = traindata, 
              y = trainlabels,
              method = "gbm", 
              trControl = fitControl,
              verbose = FALSE,
              tuneGrid = data.frame(n.trees = fit1$bestTune$n.trees,
                                    interaction.depth = fit1$bestTune$interaction.depth,
                                    shrinkage = fit1$bestTune$shrinkage,
                                    n.minobsinnode = fit1$bestTune$n.minobsinnode))

tmp <- predict(fit2, newdata = testdata, type = "prob")
tmp <- as.data.frame(tmp, row.names = NULL)
print(colnames(tmp))
print(head(tmp))
predicted[2:8] <- as.data.frame(tmp)


print("Results: ")
head(predicted)

saveRDS(predicted, file = snakemake@output[["predictions"]])

