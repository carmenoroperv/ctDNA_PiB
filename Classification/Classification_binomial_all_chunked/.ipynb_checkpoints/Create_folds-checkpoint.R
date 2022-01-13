library(tidyverse)
library(caret)
library(splitTools)


class_type <- snakemake@params[["class_type"]]
data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")


print(head(data))
print(head(sample_types))


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

message(class_type)

for (cv_rep in 1:10){
    message(paste("CV repetition number: ", cv_rep, sep = ""))
    set.seed(cv_rep)
    folds <- create_folds(data$sample_type, k = 10)
    for (fold_nr in 1:length(folds)) {
        saveRDS(folds[[fold_nr]], paste("Classification/Classification_binomial_all_chunked/Methylation_Folds/", class_type, "/Fold_", fold_nr, "_for_CV_rep_", cv_rep, "_", class_type, ".rds", sep = ""))
    }
    }