library(tidyverse)
library(pROC)
library(glmnet)
library(glmnetUtils)
library(splitTools)
library(doParallel)
library(foreach)
library(doRNG)
library(multiROC)
library(dummies)

ATAC_pred <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(ATAC_pred, sample_types, by="sample")
data %>% group_by(sample_type) %>% summarize(n = n())
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data <- data %>% droplevels("Duodenal_Cancer")
data %>% group_by(sample_type) %>% summarize(n = n())
data <- data %>% select(-sample)


get_cvm <- function(model) {
   index <- match(model$lambda.min, model$lambda)
  model$cvm[index]
}

nested_CV_lasso <- function(data, k_inner_cv, k_outer_cv){
    
    y <- data %>% dplyr::select(sample_type) %>%  as.matrix()
    X <- data %>% dplyr::select(-sample_type) %>% as.matrix() 
    observed <- y
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    return_tibble <- foreach(i = 1:k_outer_cv, 
                            .inorder = TRUE,
                            .options.RNG = 1985,
                            .combine = "rbind",
                            .packages = c("splitTools", "glmnetUtils", "tidyverse")) %dorng% { # repeated Cross-validation loop
        
        get_cvm <- function(model) {
            index <- match(model$lambda.min, model$lambda)
            model$cvm[index]
        }
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        folds <- create_folds(y, k = k_inner_cv)

        predicted <- tibble(CV_rep = rep(i, nrow(data)),
                            Bile_Duct_Cancer = rep(NA, nrow(data)),
                            Breast_Cancer = rep(NA, nrow(data)),
                            Colorectal_Cancer = rep(NA, nrow(data)),
                            Gastric_cancer = rep(NA, nrow(data)),
                            Healthy = rep(NA, nrow(data)),
                            Lung_Cancer = rep(NA, nrow(data)),
                            Ovarian_Cancer = rep(NA, nrow(data)),
                            Pancreatic_Cancer = rep(NA, nrow(data)))           

        for (fold in folds){
            message(paste("CV inner loop, CV rep number: ", i, sep = ""))
            testdata  <- X[-fold,]
            traindata <- X[fold,]
            train_y <- y[fold,]
            
            ########### nested CV to find best alpha and lambda on train folds ###########
            set.seed(0) # alpha
            lasso_cva <- cva.glmnet(traindata, train_y, nfolds = 10, family = "multinomial")
            enet_performance <- data.frame(alpha = lasso_cva$alpha)
            models <- lasso_cva$modlist
            enet_performance$cvm <- vapply(models, get_cvm, numeric(1))
            minix <- which.min(enet_performance$cvm)
            best_alpha <- lasso_cva$alpha[minix]

            set.seed(0) # lambda
            lasso_cv <- cv.glmnet(traindata, 
                                  train_y, 
                                  alpha = best_alpha, 
                                  standardize = TRUE, 
                                  nfolds = 10, 
                                  family = "multinomial")

            lambda_cv <- lasso_cv$lambda.min
            
            message("best_alpha")
            message(best_alpha)
            message("best_lambda")
            message(lambda_cv)
            ####################################################################

            fit       <- glmnet(traindata, train_y, family = "multinomial", alpha = best_alpha, lambda = lambda_cv)
            tmp       <- predict(fit, s=lambda_cv, testdata, type = "response")

            
            tmp <- as.data.frame(tmp[, , ], row.names = NULL)
            predicted[-fold, 2:9] <- as.data.frame(tmp)
        }
        
    return(predicted)
    } # end of outer cv loop
    stopCluster(cl)
    registerDoSEQ()
    
    add_observed <- tibble(observed = rep(pull(data, sample_type), k_outer_cv))
    return_tibble <- cbind(add_observed, return_tibble)
    
    return(return_tibble)
}


k_outer_cv = 10
results <- nested_CV_lasso(data, k_inner_cv = 10, k_outer_cv = k_outer_cv)

saveRDS(results, file = snakemake@output[["lasso_output"]])

print("AUC calculation with pROC package == One vs. one")


rocs <- list()
list_names <- paste0("CV_rep_", 1:k_outer_cv)
for (i in 1:k_outer_cv){
    res_CV <- results %>% filter(CV_rep == i) %>% select(-c(CV_rep, observed))
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
    res_CV <- data.frame(results %>% filter(CV_rep == i) %>% select(-c(CV_rep, observed)))
    
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
    plot_roc_df <- plot_roc_df %>% select(-Method)
    
    if (i == 1){
        Specificities <- plot_roc_df %>% select(Specificity)
        Sensitivities <- plot_roc_df %>% select(Sensitivity)
        Groups  <- plot_roc_df %>% select(Group)
    }
    else {
        Specificities <- cbind(Specificities, plot_roc_df %>% select(Specificity))
        Sensitivities <- cbind(Sensitivities, plot_roc_df %>% select(Sensitivity))
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


png(snakemake@output[["lasso_roc"]])
print(p)
dev.off()

print("Mean AUC over 10 repetitions of 10-fold CV: ")
(mean_AUCs <- colMeans(AUCs[sapply(AUCs, is.numeric)]))

results01 <- tibble()

for (i in 1:k_outer_cv){
    res_CV <- results %>% filter(CV_rep == i)
    obs_rep <- res_CV %>% select(c(CV_rep, observed))
    res_CV <- res_CV %>% select(-c(CV_rep, observed))
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