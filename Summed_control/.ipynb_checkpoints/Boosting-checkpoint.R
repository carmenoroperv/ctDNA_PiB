library(tidyverse)
library(tidyr)
library(scales)
library(LICORS)
library(slider)
library(glmnet)
library(gbm)

sum_control_ATAC_bin_rm = readRDS(snakemake@input[["input_train"]])

set.seed(0)

fit <- gbm(formula  = ATAC_val ~ ., 
               data = sum_control_ATAC_bin_rm, 
            n.trees =600, 
           cv.folds = 10)

pd <- tibble(rmse_cv = sqrt(fit$cv.error), 
             rmse_train = sqrt(fit$train.error)) %>%
  mutate(tree = row_number()) %>%
  pivot_longer(names_to = "key", values_to = "value", -tree)

ggplot(pd, aes(x=tree, y=value, color=key)) + 
  geom_line() + 
  geom_point() + 
  NULL

gbm.perf(fit, method = "cv")

fit$train.error[which.min(sqrt(fit$train.error))]

fit$cv.error[which.min(sqrt(fit$cv.error))]

predictions <- fit$cv.fitted
observed <- as.data.frame(sum_control_ATAC_bin_rm$ATAC_val)

combined <- cbind(predictions, observed)
colnames(combined) <- c("predictions", "observed")
head(combined)

p1 <- ggplot(data = combined, aes(x = observed, y = predictions)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)

ggsave(plot = p1, file = snakemake@output[["boosting_plot_summed"]])
summed_corr <- cor(combined$observed, combined$predictions)

all_individuals = readRDS(snakemake@input[["input_test"]])

ATAC = read.table(snakemake@input[["ATAC_input"]])
colnames(ATAC) = c("bin", "ATAC_val")
all_individuals_ATAC <- inner_join(all_individuals, ATAC, by ="bin") 

testdata <- all_individuals %>% select(-sample) %>% select(-bin)

y <- sum_control_ATAC_bin_rm %>% dplyr::select(ATAC_val) %>% as.matrix()
tmp <- predict(fit, testdata)
data<- cbind(tmp, y)
colnames(data) <- c("predicted", "observed")

individual_corr <- cor(data$predicted, data$observed)
p2 <- ggplot(data, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)
ggsave(plot = p2, file = snakemake@output[["boosting_plot_individual"]])

correlations <- rbind(summed_corr, individual_corr) 
rownames(correlations) <- c("summed controls", "control individually")
write.csv(correlations, snakemake@output[["lasso_ridge_corr"]]