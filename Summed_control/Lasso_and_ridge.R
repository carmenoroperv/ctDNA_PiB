library(tidyverse)
library(tidyr)
library(slider)
library(glmnet)
library(plotly)
library(Matrix)
require(methods)

bins_summed <- read.table("data/summed_controls_250kb_histograms.txt", header = TRUE)
bins_summed <- bins_summed %>% select(bin)

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

#plot(lasso_cv)

lambda_cv <- lasso_cv$lambda.min

print("Starting to run the 10-fold CV")

CV_lasso <- function(data, nfolds){
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
        
        #if (standardization == "FALSE"){ 
        #fit       <- glmnet(traindata, train_y, alpha = alpha_method, lambda = lambda_cv, standardize = FALSE)
        #    } else if (standardization == "TRUE"){ 
        #fit       <- glmnet(traindata, train_y, alpha = alpha_method, lambda = lambda_cv, standardize = TRUE)}
        
        tmp       <- predict(lasso_cv, newx = testdata, alpha = alpha_method, s="lambda.min")
        predicted[rows] <- tmp
    }
    
    rm(tmp)
    observed <- data$ATAC_val
    head(observed)
    mean(observed)
    se    <- (observed-predicted)^2
    mse   <- mean(se)
    (rmse <- sqrt(mse))
    
    return(tibble(observed = observed, predicted = predicted))
}


res_lasso_pred <- CV_lasso(sum_control_ATAC_bin_rm, 10)
print("Head of the cross-validated predictions on the summed controls")
print(head(res_lasso_pred))
res_lasso_pred <- cbind(bins_summed, res_lasso_pred)
saveRDS(res_lasso_pred, snakemake@output[["lasso_ridge_predictions_summed"]])

p1 <- ggplot(res_lasso_pred, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)


#ggsave(plot = p1, file = snakemake@output[["lasso_ridge_plot_summed"]])
png(snakemake@output[["lasso_ridge_plot_summed"]])
print(p1)
dev.off()


summed_corr = cor(res_lasso_pred$observed, res_lasso_pred$predicted)

print("Correlation between the observed ATAC values and predicted ATAC values for the summed_controls")
print(summed_corr)

#########PREDICTING ON ALL INDIVIDUALS
#all_individuals = readRDS("../data/cases_controls/cases_controls_rds_format/all_samples_normalized_CONTROLS.rds")

all_individuals = readRDS(snakemake@input[["input_test"]])

#ATAC = read.table("../data/ATACseq_250kb_bins.txt")
ATAC = read.table(snakemake@input[["ATAC_input"]], header = FALSE)
colnames(ATAC) = c("bin", "ATAC_val")
ATAC$ATAC_val <- as.character(ATAC$ATAC_val)
ATAC$ATAC_val <- as.numeric(ATAC$ATAC_val)
ATAC$bin <- as.character(ATAC$bin)
print(head(ATAC))
print(str(ATAC))

all_individuals_ATAC <- inner_join(all_individuals, ATAC, by ="bin") 

testdata <- all_individuals %>% select(-sample) %>% select(-bin)
#testdata <- Matrix(as.data.frame(testdata), sparse = TRUE)
n_rows_testdata <- dim(testdata)[1]
print("The number of rows in test data")
print(n_rows_testdata)

testdata1 <- testdata[1:round(n_rows_testdata/2), ]
testdata2 <- testdata[(round(n_rows_testdata/2)+1):n_rows_testdata, ]

print("N rows in testdata1")
print(dim(testdata1))
print("N rows in testdata2")
print(dim(testdata2))

testdata1 <- testdata1 %>% as.matrix()
testdata2 <- testdata2 %>% as.matrix()

#y <- sum_control_ATAC_bin_rm %>% dplyr::select(ATAC_val) %>% as.matrix()
#X <- sum_control_ATAC_bin_rm %>% dplyr::select(-ATAC_val) %>% as.matrix()

#if (standardization == "FALSE"){ 
#        fit       <- glmnet(X, y, alpha = alpha_method, lambda = lambda_cv, standardize = FALSE)
#            } else if (standardization == "TRUE"){ 
#        fit       <- glmnet(X, y, alpha = alpha_method, lambda = lambda_cv, standardize = TRUE)}

tmp1       <- predict(lasso_cv, newx = testdata1, alpha = alpha_method, s="lambda.min")
#tmp1       <- predict(fit, s=lambda_cv, newx = testdata1)
tmp1 <- as.data.frame(tmp1)
tmp1 <- tibble(tmp1)

tmp2       <- predict(lasso_cv,  newx = testdata2, alpha = alpha_method, s="lambda.min")
#tmp2       <- predict(fit, s=lambda_cv, newx = testdata2)
tmp2 <- as.data.frame(tmp2)
tmp2 <- tibble(tmp2)

tmp <- rbind(tmp1, tmp2)
print("Head and dimensions of the predictions for all individuals")
print(head(tmp))
print(dim(tmp))

data <- cbind(tmp, tibble(ATAC_val = all_individuals_ATAC$ATAC_val))
colnames(data) <- c("predicted", "observed")

print("Head of the predictions and observations for all individuals")
print(head(data))
print(str(data))

print("Correlation between the observed ATAC values and predicted ATAC values for all individuals")
individual_corr <- cor(data$predicted, data$observed)
print(individual_corr)



p2 <- ggplot(data, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)

#ggsave(plot = p2, file = snakemake@output[["lasso_ridge_plot_indiv"]])
png(snakemake@output[["lasso_ridge_plot_indiv"]])
print(p2)
dev.off()


correlations <- rbind(tibble(cor = summed_corr), tibble(cor = individual_corr)) 
rownames(correlations) <- c("summed controls", "all_samples")

print("Correlations on both summed_controls and all_samples")
print(correlations)


write.csv(correlations, snakemake@output[["lasso_ridge_corr"]])

## SAVE PREDICTIONS
pred <- tibble(sample = all_individuals_ATAC$sample, bin = all_individuals_ATAC$bin, ATAC_observed = data$observed, ATAC_predicted = data$predicted)
saveRDS(pred, snakemake@output[["lasso_ridge_predictions"]])

          
          
        