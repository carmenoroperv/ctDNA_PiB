library(tidyverse)
library(MASS)
library(pROC)
library(dummies)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
#, INSTALL_opts = '--no-lock'
library(splitTools)
library(multiROC)
library(doParallel)
library(foreach)
library(doRNG)
library(rngtools)

class_type <- snakemake@params[["class_type"]]
data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

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

cross_validation <- function(dataset, k_inner_cv, k_outer_cv, class_type){
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    return_tibble <- foreach(i = 1:k_outer_cv,
                                .inorder = TRUE,
                                .options.RNG = 1985,
                                .combine = "rbind",
                                .packages = c("splitTools", "MASS", "tidyverse")) %dorng% { # repeated Cross-validation loop
            
        message(class_type)
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        folds <- create_folds(dataset$sample_type, k = k_inner_cv)
        predicted <- tibble(class1_prob = rep(NA, nrow(dataset)),
                            class2_prob = rep(NA, nrow(dataset)),
                            label_pred = rep(NA, nrow(dataset)))
                                
        for (fold in folds){
            message(paste("CV inner loop, CV rep number: ", i, sep = ""))
            testdata  <- dataset[-fold,]
            testdata  <- testdata %>% dplyr::select(-sample_type)
            traindata <- dataset[fold,]

            fit       <- lda(sample_type ~ ., data=traindata, family = "binomial")
            tmp <- predict(fit, testdata)
            tmp_prob <- as.data.frame(tmp$posterior, row.names = NULL)
            predicted[-fold, 1:2] <- tmp_prob
            predicted[-fold, 3] <- tmp$class
            colnames(predicted) <- c(colnames(tmp_prob), "label_pred")
            }
        return(predicted)
    }       
    
    stopCluster(cl)
    registerDoSEQ()
    
    return_tibble <- cbind(tibble(observed = rep(observed, k_outer_cv), 
                           CV_rep = rep(1:k_outer_cv, each=nrow(dataset))), return_tibble)
                        
    return(return_tibble)
}

results <- cross_validation(data, k_inner_cv = 10, k_outer_cv = 10, class_type = class_type)

print("Results: ")
head(results)

saveRDS(results, file = snakemake@output[["predictions"]])