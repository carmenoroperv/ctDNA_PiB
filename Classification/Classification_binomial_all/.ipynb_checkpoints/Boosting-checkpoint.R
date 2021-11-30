library(tidyverse)
library(caret)
library(pROC)
library(xgboost)
library(doParallel)
library(foreach)
library(doRNG)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
#, INSTALL_opts = '--no-lock'
library(splitTools)
library(e1071)

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
    data <- data %>% mutate(sample_type = ifelse(sample_type == class_type, "Cancer", "Other"))
    } else {
    data <- data %>% mutate(sample_type = ifelse(sample_type == class_type, "Healthy", "Cancer"))
}

observed  <- data$sample_type


data$sample_type <- as.factor(data$sample_type)
print("Sample_type levels")
levels(data$sample_type)

cross_validation <- function(dataset, k_inner_cv, k_outer_cv, class_type){
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    set.seed(0)
    return_tibble <- foreach(i = 1:k_outer_cv, 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "rbind",
                            .packages = c("splitTools", "gbm", "caret", "tidyverse", "e1071")) %dorng% { # repeated Cross-validation loop
        
        message(class_type)
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        folds <- create_folds(dataset$sample_type, k = k_inner_cv)
        predicted <- rep(NA, nrow(dataset))

        for (fold in folds){
            message(paste("CV inner loop, CV rep number: ", i, sep = ""))
            testdata  <- dataset[-fold,]
            testdata <- testdata %>% select(-sample_type)
            traindata <- dataset[fold,]
            trainlabels <- traindata$sample_type
            traindata <- traindata %>% select(-sample_type)

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
            
            #if (class == "Healthy"){
            #    tmp <- tmp$Cancer
            #}
            #else{
            #    tmp <- tmp$Cancer
            #}
            
            predicted[-fold] <- tmp$Cancer
        }
        
        predicted = tibble("{class_type}_pred" := predicted)
        return(predicted)
        
    } # end of outer cv loop
    
    stopCluster(cl)
    registerDoSEQ()
    
    return_tibble <- cbind(tibble(observed = rep(observed, k_outer_cv), 
                           CV_rep = rep(1:k_outer_cv, each=nrow(dataset))), return_tibble)
    
    return(return_tibble)
}


results = cross_validation(data, k_inner_cv = 10, k_outer_cv = 10, class_type = class_type)

print("Results: ")
head(results)

saveRDS(results, file = snakemake@output[["predictions"]])
