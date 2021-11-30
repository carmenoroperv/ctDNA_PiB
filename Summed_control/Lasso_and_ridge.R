library(tidyverse)
library(tidyr)
library(slider)
library(glmnet)
library(plotly)

METHOD = as.character(snakemake@params[["METHOD"]])
standardization = as.character(snakemake@params[["STD"]])
sum_control_ATAC_bin_rm = readRDS(snakemake@input[["input_train"]])


if (METHOD == "LASSO"){
    alpha_method = 1
    } else if (METHOD == "RIDGE"){
    alpha_method = 0}



set.seed(0)
#X==predictor
#y==response

y <- sum_control_ATAC_bin_rm %>% dplyr::select(ATAC_val) %>%  as.matrix()
X <- sum_control_ATAC_bin_rm %>% dplyr::select(-ATAC_val) %>% as.matrix()

lambdas_to_try <- 10^seq(-20, -1, length.out = 100)
#standardize = FALSE -- range -10 to -5
#standardize = TRUE -- range -7 to -3


if (standardization == "TRUE"){ 
    lasso_cv <- cv.glmnet(X, y, alpha = alpha_method, lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 10)
    } else if (standardization == "FALSE"){
    lasso_cv <- cv.glmnet(X, y, alpha = alpha_method, lambda = lambdas_to_try,
                      standardize = FALSE, nfolds = 10)}

#Plotting the MSE and log(lambda)

plot(lasso_cv)

lambda_cv <- lasso_cv$lambda.min


CV_lasso<-function(data, nfolds){
    set.seed(0)
    cvfolds <- cut(1:nrow(data), breaks = nfolds, labels = F)
    cvfolds <- sample(cvfolds)
    

    predicted <- rep(NA, nrow(data))
    #data_as_matrix <- data %>% dplyr::select(-ATAC_val) %>% as.matrix() 
    
    y <- data %>% dplyr::select(ATAC_val) %>%  as.matrix()
    X <- data %>% dplyr::select(-ATAC_val) %>% as.matrix() 
    
    for (i in 1:nfolds){
        rows      <- which(cvfolds==i)
        testdata  <- X[rows,]
        traindata <- X[-rows,]
        train_y <- y[-rows,]
        
        if (standardization == "FALSE"){ 
        fit       <- glmnet(traindata, train_y, alpha = alpha_method, lambda = lambda_cv, standardize = FALSE)
            } else if (standardization == "TRUE"){ 
        fit       <- glmnet(traindata, train_y, alpha = alpha_method, lambda = lambda_cv, standardize = TRUE)}
        
        tmp       <- predict(fit, s=lambda_cv, testdata)
        predicted[rows] <- tmp
    }
    
    rm(tmp)
    observed <- y
    head(observed)
    mean(observed)
    se    <- (observed-predicted)^2
    mse   <- mean(se)
    (rmse <- sqrt(mse))
    
    return(tibble(observed = observed, predicted = predicted))
}


res_lasso_pred <- CV_lasso(sum_control_ATAC_bin_rm, 10)
#head(res_lasso_pred)

p1 <- ggplot(res_lasso_pred, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)


#ggsave(plot = p1, file = "output.png")
ggsave(plot = p1, file = snakemake@output[["lasso_ridge_plot_summed"]])

summed_corr = cor(res_lasso_pred$observed, res_lasso_pred$predicted)

summed_corr

#########PREDICTING ON ALL INDIVIDUALS
#all_individuals = readRDS("../data/cases_controls/cases_controls_rds_format/all_samples_normalized_CONTROLS.rds")

all_individuals = readRDS(snakemake@input[["input_test"]])

#ATAC = read.table("../data/ATACseq_250kb_bins.txt")
ATAC = read.table(snakemake@input[["ATAC_input"]])
colnames(ATAC) = c("bin", "ATAC_val")

all_individuals_ATAC <- inner_join(all_individuals, ATAC, by ="bin") 

testdata <- all_individuals %>% select(-sample) %>% select(-bin)


y <- sum_control_ATAC_bin_rm %>% dplyr::select(ATAC_val) %>% as.matrix()
X <- sum_control_ATAC_bin_rm %>% dplyr::select(-ATAC_val) %>% as.matrix()

if (standardization == "FALSE"){ 
        fit       <- glmnet(X, y, alpha = alpha_method, lambda = lambda_cv, standardize = FALSE)
            } else if (standardization == "TRUE"){ 
        fit       <- glmnet(X, y, alpha = alpha_method, lambda = lambda_cv, standardize = TRUE)}
        
tmp       <- predict(fit, s=lambda_cv, testdata)

tmp <- as.data.frame(tmp)

data<- cbind(tmp, y)
colnames(data) <- c("predicted", "observed")

individual_corr <- cor(data$predicted, data$observed)
individual_corr



p2 <- ggplot(data, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)

#ggsave(plot = p2, file = "output.png")
ggsave(plot = p2, file = snakemake@output[["lasso_ridge_plot_individual"]])


correlations <- rbind(summed_corr, individual_corr) 
rownames(correlations) <- c("summed controls", "control individually")


correlations


write.csv(correlations, snakemake@output[["lasso_ridge_corr"]]