library(tidyverse)
library(glmnet)
library(glmnetUtils)
library(dummies)

system("R --max-ppsize=100000 --save")
options(expressions = 500000)


data_type <- snakemake@params[["data_type"]]
class_type <- snakemake@params[["class_type"]]
data <- readRDS(snakemake@input[["input_predictions"]])
sample_types <- read.table(snakemake@input[["input_sample_types"]], header = F, sep = " ")
colnames(sample_types) <- c("sample", "sample_type")

fold <- readRDS(snakemake@input[["input_fold"]])
fold_nr <- as.numeric(snakemake@params[["fold_nr"]])
cv_rep <- as.numeric(snakemake@params[["cv_rep"]])

data <- merge(data, sample_types, by="sample")
data <- data %>% dplyr::select(-sample)

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

print("Head of fold (train rows)")
print(fold[0:6])
        
message(class_type)
message(paste("CV repetition number: ", cv_rep, sep = ""))
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

predicted <- tibble(row_predicted = test_rows,
                    second_class_prob = rep(NA, nrow(data) - length(fold)),
                    label_pred = rep(NA, nrow(data) - length(fold)))

        
get_model_params <- function(fit) {
    alpha <- fit$alpha
    lambdaMin <- sapply(fit$modlist, `[[`, "lambda.min")
    lambdaSE <- sapply(fit$modlist, `[[`, "lambda.1se")
    error <- sapply(fit$modlist, function(mod) {min(mod$cvm)})
    best <- which.min(error)
    data.frame(alpha = alpha[best], lambdaMin = lambdaMin[best],
               lambdaSE = lambdaSE[best], error = error[best])
}
        

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

# save coefficients
#if (cv_rep == 1){
#    coef <- coef(lasso_cva, alpha = best_alpha, , s="lambda.min")
#    saveRDS(coef, paste("Classification_output/", data_type, #"/Binomial_models_output_chunked/Lasso_coefficients_full_data/lasso_coefficients_CLASS", class_type, "_CVrep_", as.character(cv_rep), #"_fold_", as.character(fold_nr), ".rds", sep = ""))
#}

###############################################################################
#fit       <- glmnet(traindata, train_y, family = "binomial", alpha = best_alpha, lambda = best_lambda_min)
tmp       <- predict(lasso_cva, s="lambda.min", alpha = best_alpha, testdata, type = "response")
tmp_class <- predict(lasso_cva, s="lambda.min", alpha = best_alpha, testdata, type = "class")
predicted[2] <- tmp
predicted[3] <- tmp_class
   


print("Results: ")
head(predicted)

saveRDS(predicted, file = snakemake@output[["predictions"]])
