library(tidyverse)
library(MASS)
library(pROC)
library(dummies)
library(splitTools)
library(multiROC)
library(doParallel)
library(foreach)
library(doRNG)

data <- readRDS(snakemake@input[["input_predictions"]])

sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")


data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)
unique(data$sample_type)
data %>% group_by(sample_type) %>% summarize(n = n())
data <-subset(data, sample_type != "Duodenal_Cancer")
unique(data$sample_type)
data$sample_type <- as.factor(data$sample_type)
data = data %>% droplevels("Duodenal_Cancer")

cross_validation <- function(dataset, k_inner_cv, k_outer_cv){
    
    y_all <- dataset$sample_type
    classes <- unique(y_all)
    
    cl <- makePSOCKcluster(8, outfile="")
    registerDoParallel(cl)
    return_tibble <- foreach(class = 1:length(unique(y_all)), 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "cbind",
                            .packages = c("splitTools", "MASS", "tidyverse")) %dorng% {
        message(paste("Class: ", classes[class], sep = ""))
        y <- ifelse(y_all==classes[class], 1, 0)
        return_vector_for_class <- c()
    
        for (i in 1:k_outer_cv){ # repeated Cross-validation loop
            message(paste("CV repetition number: ", i, sep = ""))
            set.seed(i)
            folds <- create_folds(y, k = k_inner_cv)
            predicted <- rep(NA, nrow(dataset))
                                
            for (fold in folds){
                message(paste("CV inner loop, CV rep number: ", i, sep = ""))
                testdata  <- dataset[-fold,]
                testdata  <- testdata %>% dplyr::select(-sample_type)
                traindata <- dataset[fold,]
                traindata$sample_type <- ifelse(traindata$sample_type == classes[class], 1, 0)


                fit       <- lda(sample_type ~ ., data=traindata, family = "binomial")
                tmp <- predict(fit, testdata)
                tmp <- tmp$posterior[,2]
                predicted[-fold] <- tmp
                }
            
            return_vector_for_class <- c(return_vector_for_class, predicted)
        }       
        return(tibble("{classes[class]}_pred" := return_vector_for_class))
    }
    stopCluster(cl)
    registerDoSEQ()
    
    return_tibble <- cbind(tibble(observed = rep(y_all, k_outer_cv), 
                           CV_rep = rep(1:k_outer_cv, each=nrow(dataset))), return_tibble)
                        
    return(return_tibble)
}

k_outer_cv = 10
results <- cross_validation(data, k_inner_cv = 10, k_outer_cv = k_outer_cv)

saveRDS(results, file = snakemake@output[["LDA_output"]])

for (col in 1:ncol(results)){
    colnames(results)[col] <-  sub("_pred", "", colnames(results)[col])
}

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

png(snakemake@output[["LDA_roc"]])
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

head(results01)


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