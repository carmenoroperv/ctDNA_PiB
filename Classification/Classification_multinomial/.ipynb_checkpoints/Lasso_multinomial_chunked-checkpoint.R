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

#system("R --max-ppsize=100000 --save")
#options(expressions = 500000)

ATAC_pred <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")
data <- merge(ATAC_pred, sample_types, by="sample")
data <- data %>% select(-sample)

fold <- readRDS(snakemake@input[["input_fold"]])
fold_nr <- as.numeric(snakemake@params[["fold_nr"]])
cv_rep <- as.numeric(snakemake@params[["cv_rep"]])

data <- data %>% filter(sample_type != "Healthy")
print(unique(data$sample_type))
data <-subset(data, sample_type != "Duodenal_Cancer")
data$sample_type <- as.factor(data$sample_type)
data <- data %>% droplevels(c("Duodenal_Cancer", "Healthy"))

print(data %>% group_by(sample_type) %>% summarize(n = n()))
print("Sample_type levels")
print(levels(data$sample_type))

observed  <- data$sample_type

print("Head of fold (train rows)")
print(fold[0:6])
        
print(paste("CV repetition number: ", cv_rep, sep = ""))
set.seed(cv_rep)

    
y <- data %>% dplyr::select(sample_type) %>%  as.matrix()
X <- data %>% dplyr::select(-sample_type) %>% as.matrix() 

testdata  <- X[-fold,]
traindata <- X[fold,]
train_y <- y[fold,]

rows <- data %>% mutate(row_name = row_number()) %>% dplyr::select(row_name)
test_rows <- rows[-fold,]
print("Head of test fold (test_rows)")
print(head(test_rows))
        
get_model_params <- function(fit) {
      alpha <- fit$alpha
      lambdaMin <- sapply(fit$modlist, `[[`, "lambda.min")
      lambdaSE <- sapply(fit$modlist, `[[`, "lambda.1se")
      error <- sapply(fit$modlist, function(mod) {min(mod$cvm)})
      best <- which.min(error)
      data.frame(alpha = alpha[best], lambdaMin = lambdaMin[best],
                 lambdaSE = lambdaSE[best], error = error[best])
    }
        

predicted <- tibble(row_predicted = test_rows,
                    Bile_Duct_Cancer = rep(NA, nrow(data) - length(fold)),
                    Breast_Cancer = rep(NA, nrow(data) - length(fold)),
                    Colorectal_Cancer = rep(NA, nrow(data) - length(fold)),
                    Gastric_cancer = rep(NA, nrow(data) - length(fold)),
                    Lung_Cancer = rep(NA, nrow(data) - length(fold)),
                    Ovarian_Cancer = rep(NA, nrow(data) - length(fold)),
                    Pancreatic_Cancer = rep(NA, nrow(data) - length(fold)))           


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

#fit       <- glmnet(traindata, train_y, family = "multinomial", alpha = best_alpha, lambda = best_lambda_min)
tmp       <- predict(lasso_cva, s="lambda.min", alpha = best_alpha, testdata, type = "response")



tmp <- as.data.frame(tmp[, , ], row.names = NULL)
print(colnames(tmp))
print(head(tmp))
print("Starting to write the predictions")
print("Dimensions of predictions for this fold")
print(dim(tmp))
print("Dimensions of predicted tibble")
predicted[2:8] <- as.data.frame(tmp)

print("Results: ")
head(predicted)

saveRDS(predicted, file = snakemake@output[["predictions"]])
