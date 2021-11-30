library(tidyverse)
library(MASS)
library(pROC)
library(multiROC)
library(dummies)
library(doParallel)
library(foreach)
library(doRNG)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
library(splitTools)

data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)

data <- data %>% filter(sample_type != "Healthy")
message(unique(data$sample_type))
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data <- data %>% droplevels(c("Duodenal_Cancer", "Healthy"))

message(data %>% group_by(sample_type) %>% summarize(n = n()))
message(levels(data$sample_type))

observed  <- data$sample_type

cross_validation <- function(dataset, k_inner_cv, k_outer_cv){
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    return_tibble <- foreach(i = 1:k_outer_cv, 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "rbind",
                            .packages = c("splitTools", "MASS", "tidyverse")) %dorng% { # repeated Cross-validation loop
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        cvfolds <- cut(seq_len(nrow(dataset)), breaks = k_inner_cv, labels = F)
        cvfolds <- sample(cvfolds)

        predicted <- tibble(CV_rep = rep(i, nrow(dataset)),
                    Bile_Duct_Cancer = rep(NA, nrow(dataset)),
                    Breast_Cancer = rep(NA, nrow(dataset)),
                    Colorectal_Cancer = rep(NA, nrow(dataset)),
                    Gastric_cancer = rep(NA, nrow(dataset)),
                    Lung_Cancer = rep(NA, nrow(dataset)),
                    Ovarian_Cancer = rep(NA, nrow(dataset)),
                    Pancreatic_Cancer = rep(NA, nrow(dataset)))  

        for (n in 1:k_inner_cv){
            message(paste("CV inner loop, number: ", n, sep = ""))
            rows      <- which(cvfolds==n)
            testdata  <- dataset[rows,]
            testdata <- testdata %>% dplyr::select(-sample_type)
            traindata <- dataset[-rows,]


            fit       <- lda(sample_type ~ ., data=traindata, family = "multinomial")
            tmp <- predict(fit, testdata)
            tmp <- as.data.frame(tmp$posterior, row.names = NULL)
            message(colnames(tmp))
            message(head(tmp))
            predicted[rows, 2:8] <- as.data.frame(tmp)
            
            }
        
    return(predicted)
    } # end of outer cv loop
    
    stopCluster(cl)
    registerDoSEQ()

    return_tibble <- cbind(tibble(observed = rep(observed, k_outer_cv), return_tibble))
                        
    return(return_tibble)
}

k_outer_cv = 10
results <- cross_validation(data, k_inner_cv = 10, k_outer_cv = k_outer_cv)

print("Results: ")
head(results)

saveRDS(results, file = snakemake@output[["predictions"]])

