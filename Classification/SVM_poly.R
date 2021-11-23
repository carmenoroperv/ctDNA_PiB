library(tidyverse)
library(caret)
library(pROC)
install.packages("kernlab", repos = "http://cran.us.r-project.org")
library(kernlab)
library(doParallel)
library(foreach)
library(doRNG)

ATAC_pred <-  readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")

colnames(sample_types) <- c("sample", "sample_type")
head(sample_types)

data <- merge(ATAC_pred, sample_types, by="sample")
data <- data %>% mutate(sample_type01 = ifelse(sample_type == "Healthy", "Healthy", "Cancer"))
data <- data %>% select(-sample_type)
data <- data %>% select(-sample)
data$sample_type01 <- as.factor(data$sample_type01)
levels(data$sample_type01)

cross_validation <- function(dataset, k_inner_cv, k_outer_cv){
    
    observed  <- dataset$sample_type01
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    
    return_tibble <- foreach(i = 1:k_outer_cv, 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "cbind",
                            .packages = c("kernlab", "caret", "tidyverse")) %dorng% { # repeated Cross-validation loop
        
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        cvfolds <- cut(seq_len(nrow(dataset)), breaks = k_inner_cv, labels = F)
        cvfolds <- sample(cvfolds)
        predicted <- rep(NA, nrow(dataset))

            for (n in 1:k_inner_cv){
                message(paste("CV inner loop (CV fold) number: ", n, sep = ""))
                rows      <- which(cvfolds==n)
                testdata  <- dataset[rows,]
                testlabels <- testdata$sample_type01
                testdata <- testdata %>% select(-sample_type01)

                traindata <- dataset

                traindata <- dataset[-rows,]

                trainlabels <- traindata$sample_type01


                ################# Nested cross validation #######################
                set.seed(0)
                seeds <- vector(mode = "list", length = 11)
                for(i in 1:10) seeds[[i]]<- sample.int(n=2000, 1200)
                #for the last model
                seeds[[11]]<-sample.int(1000, 1)

                trControl_svm <- trainControl(method = "repeatedcv", 
                                              seeds = seeds,
                                              number = 10, 
                                              repeats = 1, 
                                              classProbs = TRUE)

                fit <- train(sample_type01 ~ .,
                             data = traindata, 
                             method = "svmPoly",
                             tuneLength = 5,
                             trControl = trControl_svm,
                             preProc = c("center", "scale"),
                             verbose=F)

                message("besttune C")
                message(fit$bestTune$C)
                message("besttune degree")
                message(fit$bestTune$degree)
                message("besttune scale")
                message(fit$bestTune$scale)
                #################################################################

                fitControl <- trainControl(classProbs = TRUE)
                fit2 <- train(sample_type01 ~ .,
                             data = traindata,
                             method =  "svmPoly",
                             trControl = fitControl,
                             verbose = FALSE,
                             tuneGrid = data.frame(C = fit$bestTune$C, 
                                                   degree = fit$bestTune$degree, 
                                                   scale = fit$bestTune$scale),
                             preProc = c("center", "scale"))
                predicted[rows] <- predict(fit2, newdata = testdata, type = "prob")[,2]

        }
        current_round_tibble <- tibble(predicted = predicted)
        return(current_round_tibble)
        } # end of outer cv loop
    
    stopCluster(cl)
    registerDoSEQ()
    
    return_tibble <- cbind(tibble(observed = observed), return_tibble)
    return(return_tibble)
}

results <- cross_validation(data, k_inner_cv = 10, k_outer_cv = 10)

results <- tibble(results, .name_repair = "unique")
head(results)

saveRDS(results, file = snakemake@output[["SVM_output"]])

# Create multiple curves to plot
rocs <- roc(observed ~ predicted...2 + 
                    predicted...3 + 
                    predicted...4 + 
                    predicted...5 + 
                    predicted...6 +
                    predicted...7 +
                    predicted...8 + 
                    predicted...9 + 
                    predicted...10 + 
                    predicted...11, data = results)

p <- ggroc(rocs)
png(file = snakemake@output[["SVM_roc"]])
print(p)
dev.off()

roc1 <- roc(results$observed, results$predicted...2)
roc2 <- roc(results$observed, results$predicted...3)
roc3 <- roc(results$observed, results$predicted...4)
roc4 <- roc(results$observed, results$predicted...5)
roc5 <- roc(results$observed, results$predicted...6)
roc6 <- roc(results$observed, results$predicted...7)
roc7 <- roc(results$observed, results$predicted...8)
roc8 <- roc(results$observed, results$predicted...9)
roc9 <- roc(results$observed, results$predicted...10)
roc10 <- roc(results$observed, results$predicted...11)

paste("Mean AUC over 10 repetitions of 10-fold CV: ", mean(c(auc(roc1), auc(roc2), auc(roc3), auc(roc4), auc(roc5), auc(roc6), auc(roc7), auc(roc8), auc(roc9), auc(roc10))), sep = "")

results_01 <- results %>% mutate(predicted...2_01 = ifelse(predicted...2 > 0.5, 1, 0), 
                                 predicted...3_01 = ifelse(predicted...3 > 0.5, 1, 0), 
                                 predicted...4_01 = ifelse(predicted...4 > 0.5, 1, 0),
                                 predicted...5_01 = ifelse(predicted...5 > 0.5, 1, 0),
                                 predicted...6_01 = ifelse(predicted...6 > 0.5, 1, 0),
                                 predicted...7_01 = ifelse(predicted...7 > 0.5, 1, 0),
                                 predicted...8_01 = ifelse(predicted...8 > 0.5, 1, 0),
                                 predicted...9_01 = ifelse(predicted...9 > 0.5, 1, 0),
                                 predicted...10_01 = ifelse(predicted...10 > 0.5, 1, 0),
                                 predicted...11_01 = ifelse(predicted...11 > 0.5, 1, 0))

error_rates <- c(mean(results_01$observed != results_01$predicted...2_01),
               mean(results_01$observed != results_01$predicted...3_01), 
               mean(results_01$observed != results_01$predicted...4_01),
               mean(results_01$observed != results_01$predicted...5_01),
               mean(results_01$observed != results_01$predicted...6_01),
               mean(results_01$observed != results_01$predicted...7_01),
               mean(results_01$observed != results_01$predicted...8_01),
               mean(results_01$observed != results_01$predicted...9_01), 
               mean(results_01$observed != results_01$predicted...10_01), 
               mean(results_01$observed != results_01$predicted...11_01))

error_rates

paste("Mean error rate over 10 repetitions of 10-fold CV: ", mean(error_rates), sep = "")

accuracies <- c(mean(results_01$observed == results_01$predicted...2_01),
               mean(results_01$observed == results_01$predicted...3_01), 
               mean(results_01$observed == results_01$predicted...4_01),
               mean(results_01$observed == results_01$predicted...5_01),
               mean(results_01$observed == results_01$predicted...6_01),
               mean(results_01$observed == results_01$predicted...7_01),
               mean(results_01$observed == results_01$predicted...8_01),
               mean(results_01$observed == results_01$predicted...9_01), 
               mean(results_01$observed == results_01$predicted...10_01), 
               mean(results_01$observed == results_01$predicted...11_01))

accuracies
paste("Mean accuracy over 10 repetitions of 10-fold CV: ", mean(accuracies), sep = "")