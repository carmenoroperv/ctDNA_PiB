library(tidyverse)
library(MASS)
library(dummies)

system("R --max-ppsize=100000 --save")
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

fit       <- lda(x = traindata, grouping = trainlabels, family = "binomial")
tmp <- predict(fit, testdata)
tmp_prob <- as.data.frame(tmp$posterior, row.names = NULL)
predicted[2:3] <- tmp_prob
predicted[4] <- tmp$class
colnames(predicted) <- c("dataset_row_no", colnames(tmp_prob), "label_pred")


print("Results: ")
head(predicted)

saveRDS(predicted, file = snakemake@output[["predictions"]])