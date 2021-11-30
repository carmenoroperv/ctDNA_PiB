library(tidyverse)
library(pROC)
library(glmnet)
library(glmnetUtils)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
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
data <- data %>% select(-sample)

data <- data %>% filter(sample_type != "Healthy")
message(unique(data$sample_type))
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data <- data %>% droplevels(c("Duodenal_Cancer", "Healthy"))

message(data %>% group_by(sample_type) %>% summarize(n = n()))
message(levels(data$sample_type))


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
        
        get_model_params <- function(fit) {
              alpha <- fit$alpha
              lambdaMin <- sapply(fit$modlist, `[[`, "lambda.min")
              lambdaSE <- sapply(fit$modlist, `[[`, "lambda.1se")
              error <- sapply(fit$modlist, function(mod) {min(mod$cvm)})
              best <- which.min(error)
              data.frame(alpha = alpha[best], lambdaMin = lambdaMin[best],
                         lambdaSE = lambdaSE[best], error = error[best])
            }
        
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        folds <- create_folds(y, k = k_inner_cv)

        predicted <- tibble(CV_rep = rep(i, nrow(data)),
                            Bile_Duct_Cancer = rep(NA, nrow(data)),
                            Breast_Cancer = rep(NA, nrow(data)),
                            Colorectal_Cancer = rep(NA, nrow(data)),
                            Gastric_cancer = rep(NA, nrow(data)),
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
            best_params <- get_model_params(lasso_cva)
            best_alpha <- best_params$alpha
            best_lambda_min <- best_params$lambdaMin 
            
            message("best_alpha")
            message(best_alpha)
            message("best_lambda")
            message(best_lambda_min)
            ####################################################################

            fit       <- glmnet(traindata, train_y, family = "multinomial", alpha = best_alpha, lambda = best_lambda_min)
            tmp       <- predict(fit, s=best_lambda_min, testdata, type = "response")

            
            tmp <- as.data.frame(tmp[, , ], row.names = NULL)
            message(colnames(tmp))
            message(head(tmp))
            predicted[-fold, 2:8] <- as.data.frame(tmp)
        }
        
    return(predicted)
    } # end of outer cv loop
    stopCluster(cl)
    registerDoSEQ()
    
    return_tibble <- cbind(tibble(observed = rep(data$sample_type, k_outer_cv), return_tibble))
    
    return(return_tibble)
}


k_outer_cv = 10
results <- nested_CV_lasso(data, k_inner_cv = 10, k_outer_cv = k_outer_cv)

print("Results: ")
head(results)

saveRDS(results, file = snakemake@output[["predictions"]])

