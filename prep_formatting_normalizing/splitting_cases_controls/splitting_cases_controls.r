library(tidyverse)


filename = snakemake@input[["input_data"]]
data <- readRDS(filename)


sample_types <- read.table(snakemake@input[["input_samples"]])
head(sample_types)


healthy_individuals <- sample_types %>% filter(V2 == "Healthy") %>% select(V1)
head(healthy_individuals)
cancer_individuals <- sample_types %>% filter(V2 != "Healthy") %>% select(V1)
head(cancer_individuals)


data_controls <- data %>% filter(sample %in% healthy_individuals$V1)
head(data_controls)
dim(data_controls)
saveRDS(data_controls, file = snakemake@output[["output_controls_normalized_trimmed"]])

data_cases <- data %>% filter(sample %in% cancer_individuals$V1)
head(data_cases)
dim(data_cases)
saveRDS(data_cases, file = snakemake@output[["output_cases_normalized_trimmed"]])