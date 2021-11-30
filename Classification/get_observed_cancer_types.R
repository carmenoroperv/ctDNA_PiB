library(tidyverse)

#data <- readRDS("../data/ATAC_predictions_train_20_predict_80/Full_data_ATAC_pred_lasso_formatted_standardized.rds")
data <- readRDS(snakemake@input[["input_predictions"]]))

sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(data, sample_types, by="sample")
data <- data %>% select(-sample)
data <- data %>% select(sample_type)
data <- data %>% filter(sample_type != "Healthy")
data %>% group_by(sample_type) %>% summarize(n = n())
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data = data %>% droplevels("Duodenal_Cancer")

head(data)
observed  <- data$sample_type

saveRDS(observed, file = snakemake@output[["observed"]])