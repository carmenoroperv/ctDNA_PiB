library(tidyverse)
library(pROC)
library(glmnet)
library(glmnetUtils)
install.packages("splitTools", repos = "http://cran.us.r-project.org/src/contrib/splitTools_0.3.1.tar.gz")
#, INSTALL_opts = '--no-lock'
library(splitTools)
library(multiROC)
library(dummies)
library(ROCR)
library(rlist)
library(doParallel)
library(foreach)
library(doRNG)
library(rngtools)

class_type <- snakemake@params[["class_type"]]
data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

data <- merge(data, sample_types, by="sample")
data <- data %>% select(-sample)

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

nested_CV_lasso <- function(dataset, k_inner_cv, k_outer_cv, class_type){
    
    y <- dataset %>% dplyr::select(sample_type) %>%  as.matrix()
    X <- dataset %>% dplyr::select(-sample_type) %>% as.matrix() 
    
    cl <- makePSOCKcluster(10, outfile="")
    registerDoParallel(cl)
    set.seed(0)
    return_tibble <- foreach(i = 1:k_outer_cv,
                                .inorder = TRUE,
                                .options.RNG = 1985,
                                .combine = "rbind",
                                .packages = c("splitTools", "glmnetUtils", "tidyverse")) %dorng% { # repeated Cross-validation
        
        get_model_params <- function(fit) {
            alpha <- fit$alpha
            lambdaMin <- sapply(fit$modlist, `[[`, "lambda.min")
            lambdaSE <- sapply(fit$modlist, `[[`, "lambda.1se")
            error <- sapply(fit$modlist, function(mod) {min(mod$cvm)})
            best <- which.min(error)
            data.frame(alpha = alpha[best], lambdaMin = lambdaMin[best],
                       lambdaSE = lambdaSE[best], error = error[best])
        }
        
        message(class_type)
        message(paste("CV repetition number: ", i, sep = ""))
        set.seed(i)
        folds <- create_folds(y, k = k_inner_cv)
        predicted <- tibble(class_prob = rep(NA, nrow(dataset)),
                            label_pred = rep(NA, nrow(dataset)))
        
        for (fold in folds){
            message(paste("CV inner loop, CV rep number: ", i, sep = ""))
            testdata  <- X[-fold,]
            traindata <- X[fold,]
            train_y <- y[fold,]

            ########### nested CV to find best alpha and lambda on train folds ###########
            set.seed(0) # alpha
            lasso_cva <- cva.glmnet(traindata, train_y, nfolds = 10, family = "binomial")
            best_params <- get_model_params(lasso_cva)
            best_alpha <- best_params$alpha
            best_lambda_min <- best_params$lambdaMin
            
            message("best_alpha")
            message(best_alpha)
            message("best_lambda")
            message(best_lambda_min)
            ###############################################################################
            fit       <- glmnet(traindata, train_y, family = "binomial", alpha = best_alpha, lambda = best_lambda_min)
            tmp       <- predict(fit, s=best_lambda_min, testdata, type = "response")
            tmp_class <- predict(fit, s=best_lambda_min, testdata, type = "class")
            predicted[-fold, 1] <- tmp
            predicted[-fold, 2] <- tmp_class
        }
        return(predicted)
    } # end of outer cv loop
    
    stopCluster(cl)
    registerDoSEQ()
    
    return_tibble <- cbind(tibble(observed = rep(observed, k_outer_cv), 
                           CV_rep = rep(1:k_outer_cv, each=nrow(dataset))), return_tibble)
    
    return(return_tibble)
}

results <- nested_CV_lasso(data, k_inner_cv = 10, k_outer_cv = 10, class_type = class_type)

print("Results: ")
head(results)

saveRDS(results, file = snakemake@output[["predictions"]])
