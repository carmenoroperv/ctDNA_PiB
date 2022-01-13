library(tidyverse)
library(caret)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
library(splitTools)
library(e1071)

system("R --max-ppsize=300000 --save")
options(expressions = 500000)

class_type <- snakemake@params[["class_type"]]
data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

fold <- readRDS(snakemake@input[["input_fold"]])
cv_rep <- as.numeric(snakemake@params[["cv_rep"]])

data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)

if (class_type != "Healthy"){
    data <- data %>% filter(sample_type != "Healthy")
    message(unique(data$sample_type))
    data %>% group_by(sample_type) %>% summarize(n = n())
    data <-subset(data, sample_type != "Duodenal_Cancer")
    data$sample_type <- as.factor(data$sample_type)
    data = data %>% droplevels("Duodenal_Cancer")
    data <- data %>% mutate(sample_type = ifelse(sample_type == class_type, class_type, "Other"))
    } else {
    data <- data %>% mutate(sample_type = ifelse(sample_type == class_type, class_type, "Cancer"))
}

observed  <- data$sample_type


data$sample_type <- as.factor(data$sample_type)
print("Sample_type levels")
levels(data$sample_type)

print("Head of fold (train rows)")
print(fold[0:6])


        
message(class_type)
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
                    class1_prob = rep(NA, nrow(data) - length(fold)),
                    class2_prob = rep(NA, nrow(data) - length(fold)),
                    label_pred = rep(NA, nrow(data) - length(fold)))


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
tmp_class <- predict(fit2, newdata = testdata)
predicted[2:3] <- tmp
predicted[4] <- tmp_class
colnames(predicted) <- c("dataset_row_no", colnames(tmp), "label_pred")

print("Results: ")
head(predicted)

saveRDS(predicted, file = snakemake@output[["predictions"]])
