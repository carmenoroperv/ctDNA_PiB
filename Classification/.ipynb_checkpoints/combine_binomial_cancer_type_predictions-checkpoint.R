library(tidyverse)

observed <- readRDS(snakemake@input[["observed"]]))

model <- snakemake@params[["model"]]

classes = c("Bile_Duct_Cancer", 
           "Breast_Cancer", 
           "Colorectal_Cancer", 
           "Gastric_cancer", 
           "Lung_Cancer", 
           "Ovarian_Cancer", 
           "Pancreatic_Cancer")

k_outer_cv = 10

combined_pred <- tibble(observed = rep(observed, k_outer_cv), 
                        CV_rep = rep(1:k_outer_cv, each=nrow(observed)))

combined_pred_PCA <- tibble(observed = rep(observed, k_outer_cv), 
                            CV_rep = rep(1:k_outer_cv, each=nrow(observed)))

for (type in 1:length(classes)){
    data_cancer_type <- readRDS(paste("../Classification_output/Binomial_models_output/", model, "_Predictions_Full_data_", classes[type], ".rds", sep = ""))
    data_cancer_type <- data_cancer_type %>% select("{class_type}_pred")
    colnames(data_cancer_type) = paste(classes[type], "pred", sep = "_")
    combined_pred <- cbind(combined_pred, data_cancer_type)
    
    data_cancer_type_PCA <- readRDS(paste("../Classification_output/Binomial_models_output/", model, "_Predictions_PCA_", classes[type], ".rds", sep = ""))
    data_cancer_type_PCA <- data_cancer_type_PCA %>% select("{class_type}_pred")
    colnames(data_cancer_type_PCA) = paste(classes[type], "pred", sep = "_")
    combined_pred_PCA <- cbind(combined_pred_PCA, data_cancer_type_PCA)
}

IRdisplay::display(head(combined_pred))
saveRDS(combined_pred, file = snakemake@output[["cancer_type_predictions"]])

IRdisplay::display(head(combined_pred_PCA))
saveRDS(combined_pred_PCA, file = snakemake@output[["cancer_type_predictions_PCA"]])