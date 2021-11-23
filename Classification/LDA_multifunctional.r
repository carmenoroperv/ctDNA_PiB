library(tidyverse)
library(MASS)
library(pROC)
library(dummies)
library(splitTools)
library(multiROC)
library(doParallel)
library(foreach)
library(doRNG)
library(rngtools)

class <- snakemake@params[["class"]]

data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)
unique(data$sample_type)

if (class != "Healthy"){
    data %>% group_by(sample_type) %>% summarize(n = n())
    data <-subset(data, sample_type != "Duodenal_Cancer")
    unique(data$sample_type)
    data$sample_type <- as.factor(data$sample_type)
    data = data %>% droplevels("Duodenal_Cancer")
    }
    
    
return_tibble <- foreach(i = 1:k_outer_cv, 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "cbind",
                            .packages = c("kernlab", "caret", "tidyverse")) %dorng% { # repeated Cross-validation loop