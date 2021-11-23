library(tidyverse)
library(caret)
library(pROC)
library(xgboost)
library(doParallel)
library(foreach)
library(doRNG)
library(multiROC)
library(dummies)

data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)
data %>% group_by(sample_type) %>% summarize(n = n())
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data <- data %>% droplevels("Duodenal_Cancer")

data$sample_type <- as.factor(data$sample_type)
levels(data$sample_type)


cross_validation <- function(dataset, k_inner_cv, k_outer_cv){
    
    observed  <- dataset$sample_type
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    return_tibble <- foreach(i = 1:k_outer_cv, 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "rbind",
                            .packages = c("gbm", "caret", "tidyverse")) %dorng% { # repeated Cross-validation loop
        
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        cvfolds <- cut(seq_len(nrow(dataset)), breaks = k_inner_cv, labels = F)
        cvfolds <- sample(cvfolds)

        predicted <- tibble(CV_rep = rep(i, nrow(dataset)),
                     Bile_Duct_Cancer = rep(NA, nrow(dataset)),
                     Breast_Cancer = rep(NA, nrow(dataset)),
                     Colorectal_Cancer = rep(NA, nrow(dataset)),
                     Gastric_cancer = rep(NA, nrow(dataset)),
                     Healthy = rep(NA, nrow(dataset)),
                     Lung_Cancer = rep(NA, nrow(dataset)),
                     Ovarian_Cancer = rep(NA, nrow(dataset)),
                     Pancreatic_Cancer = rep(NA, nrow(dataset)))

        for (n in 1:k_inner_cv){
            
            message(paste("CV inner loop (CV fold) number: ", n, sep = ""))
            rows      <- which(cvfolds==n)
            testdata  <- dataset[rows,]
            testlabels <- testdata$sample_type
            testdata <- testdata %>% select(-sample_type)

            traindata <- dataset[-rows,]
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

            fitControl <- trainControl()
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
            tmp <- as.data.frame(tmp, row.names = NULL)
            predicted[rows, 2:9] <- as.data.frame(tmp)
        }

        return(predicted)
        } # end of outer cv loop
    
    stopCluster(cl)
    registerDoSEQ()
    
    add_observed <- tibble(observed = rep(observed, k_outer_cv))
    return_tibble <- cbind(add_observed, return_tibble)
                        
    return(return_tibble)
}


k_outer_cv = 10
results = cross_validation(data, k_inner_cv = 10, k_outer_cv = k_outer_cv)

saveRDS(results, file = snakemake@output[["boosting_output"]])
head(results)

print("AUC calculation with pROC package == One vs. one")

rocs <- list()
list_names <- paste0("CV_rep_", 1:k_outer_cv)
for (i in 1:k_outer_cv){
    res_CV <- results %>% filter(CV_rep == i) %>% dplyr::select(-c(CV_rep, observed))
    roc <- multiclass.roc(response = data$sample_type, predictor = res_CV)
    rocs[[i]] <- roc
}
names(rocs) <- c(list_names)

paste0("The AUC of the first CV repetition: ", rocs[[1]]$auc)

aucs <- c()
for (i in 1:length(rocs)){
    aucs <- c(aucs, rocs[[i]]$auc) 
}

paste0("The mean AUC of all the CV repetitions: ", mean(aucs))

print("Run the multiROC and plotting for all the CV repetitions")
Specificities <- NULL
Sensitivities <- NULL

for (i in 1:k_outer_cv){
    res_CV <- data.frame(results %>% filter(CV_rep == i) %>% dplyr::select(-c(CV_rep, observed)))
    
    colnames(res_CV) <- paste(colnames(res_CV), "_pred_lasso", sep = "")

    true_label <- dummies::dummy(data$sample_type, sep = ".")
    true_label <- data.frame(true_label)
    colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
    colnames(true_label) <- paste(colnames(true_label), "_true", sep = "")
    final_df <- cbind(true_label, res_CV)

    roc_res <- multi_roc(final_df, force_diag=T)
    
    if (i == 1){
        AUCs <- as.data.frame(t(unlist(roc_res$AUC)))
    }
    else {
        AUCs <- rbind(AUCs, as.data.frame(t(unlist(roc_res$AUC))))
    }
    
    plot_roc_df <- plot_roc_data(roc_res)
    plot_roc_df <- plot_roc_df %>% dplyr::select(-Method)
    
    if (i == 1){
        Specificities <- plot_roc_df %>% dplyr::select(Specificity)
        Sensitivities <- plot_roc_df %>% dplyr::select(Sensitivity)
        Groups  <- plot_roc_df %>% dplyr::select(Group)
    }
    else {
        Specificities <- cbind(Specificities, plot_roc_df %>% dplyr::select(Specificity))
        Sensitivities <- cbind(Sensitivities, plot_roc_df %>% dplyr::select(Sensitivity))
    }
    
    #IRdisplay::display(all_plot_roc_df)

}
Specificities <- tibble(Specificities, .name_repair = "unique")
Specificities <- Specificities %>% mutate(mean = rowMeans(across()))
Sensitivities <- tibble(Sensitivities, .name_repair = "unique")
Sensitivities <- Sensitivities %>% mutate(mean = rowMeans(across()))

(AUCs <- tibble(AUCs))

all_sensitivity_specificity <- tibble(Specificity = Specificities$mean, Sensitivity = Sensitivities$mean, Group = Groups$Group)
head(all_sensitivity_specificity)

p <- ggplot(all_sensitivity_specificity, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group), size=1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
                        colour='grey', linetype = 'dotdash') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
                 legend.justification=c(1, 0), legend.position=c(.95, .05),
                 legend.title=element_blank(), 
                 legend.background = element_rect(fill=NULL, size=0.5, 
                                                           linetype="solid", colour ="black"))


png(snakemake@output[["boosting_roc"]])
print(p)
dev.off()


print("Mean AUC over 10 repetitions of 10-fold CV: ")
(mean_AUCs <- colMeans(AUCs[sapply(AUCs, is.numeric)]))


results01 <- tibble()

for (i in 1:k_outer_cv){
    res_CV <- results %>% filter(CV_rep == i)
    obs_rep <- res_CV %>% dplyr::select(c(CV_rep, observed))
    res_CV <- res_CV %>% dplyr::select(-c(CV_rep, observed))
    res_CV <- res_CV %>% mutate(pred01 = factor(colnames(res_CV)[apply(res_CV,1,which.max)], ordered = TRUE))
    res_CV <- cbind(res_CV, obs_rep)
    results01 <- rbind(results01, res_CV)
}



error_rates <- c()
accuracies <- c()
for (i in 1:k_outer_cv){
    res_CV <- results01 %>% filter(CV_rep == i)
    error_rates <- c(error_rates, mean(as.character(res_CV$observed) != as.character(res_CV$pred01)))
    accuracies <- c(accuracies, mean(as.character(res_CV$observed) == as.character(res_CV$pred01)))
}

error_rates
accuracies

paste("Mean error rate over 10 repetitions of 10-fold CV: ", mean(error_rates), sep = "")
paste("Mean accuracy over 10 repetitions of 10-fold CV: ", mean(accuracies), sep = "")