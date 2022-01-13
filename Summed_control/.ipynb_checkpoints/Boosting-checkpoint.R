library(tidyverse)
library(tidyr)
library(scales)
library(LICORS)
library(slider)
library(glmnet)
library(gbm)

bins_summed <- read.table("data/summed_controls_250kb_histograms.txt", header = TRUE)
bins_summed <- bins_summed %>% select(bin)

sum_control_ATAC_bin_rm = readRDS(snakemake@input[["input_train"]])

set.seed(0)

fit <- gbm(formula  = ATAC_val ~ ., 
               data = sum_control_ATAC_bin_rm, 
            n.trees = 600, 
           cv.folds = 10)

pd <- tibble(rmse_cv = sqrt(fit$cv.error), 
             rmse_train = sqrt(fit$train.error)) %>%
  mutate(tree = row_number()) %>%
  pivot_longer(names_to = "key", values_to = "value", -tree)

#ggplot(pd, aes(x=tree, y=value, color=key)) + 
#  geom_line() + 
#  geom_point() + 
#  NULL

print(gbm.perf(fit, method = "cv"))

print(fit$train.error[which.min(sqrt(fit$train.error))])

print(fit$cv.error[which.min(sqrt(fit$cv.error))])

predictions <- tibble(predictions = fit$cv.fitted)

combined <- cbind(predictions, tibble(observed = sum_control_ATAC_bin_rm$ATAC_val))
colnames(combined) <- c("predicted", "observed")
print("Observed ATAC values and CV predicted ATAC values for summed controls")
print(head(combined))
print(str(combined))
combined <- cbind(bins_summed, combined)
saveRDS(combined, snakemake@output[["boosting_predictions_summed"]])

p1 <- ggplot(data = combined, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)

#ggsave(plot = p1, file = snakemake@output[["boosting_plot_summed"]])
png(snakemake@output[["boosting_plot_summed"]])
print(p1)
dev.off()

summed_corr <- cor(combined$observed, combined$predicted)

all_individuals = readRDS(snakemake@input[["input_test"]])

ATAC = read.table(snakemake@input[["ATAC_input"]])
colnames(ATAC) = c("bin", "ATAC_val")
ATAC$ATAC_val <- as.character(ATAC$ATAC_val)
ATAC$ATAC_val <- as.numeric(ATAC$ATAC_val)
ATAC$bin <- as.character(ATAC$bin)
all_individuals_ATAC <- inner_join(all_individuals, ATAC, by ="bin") 

testdata <- all_individuals %>% select(-sample) %>% select(-bin)

y <- sum_control_ATAC_bin_rm %>% dplyr::select(ATAC_val) %>% as.matrix()
tmp <- predict(fit, testdata)
data <- cbind(tibble(tmp), tibble(all_individuals_ATAC$ATAC_val))
colnames(data) <- c("predicted", "observed")

individual_corr <- cor(data$predicted, data$observed)
p2 <- ggplot(data, aes(x = observed, y = predicted)) + 
    geom_point(size = 0.5) + 
    geom_smooth(method = "lm", formula = y~x)

#ggsave(plot = p2, file = snakemake@output[["boosting_plot_indiv"]])
png(snakemake@output[["boosting_plot_indiv"]])
print(p2)
dev.off()

correlations <- rbind(summed_corr, individual_corr) 
rownames(correlations) <- c("summed controls", "control individually")

print("Correlations on both summed_controls and all_samples")
print(correlations)
write.csv(correlations, snakemake@output[["boosting_corr"]])

# SAVE PREDICTIONS
pred <- tibble(sample = all_individuals_ATAC$sample, bin = all_individuals_ATAC$bin, ATAC_observed = data$observed, ATAC_predicted = data$predicted)
saveRDS(pred, snakemake@output[["boosting_predictions"]])
