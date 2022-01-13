library(tidyverse)

obs <- readRDS(snakemake@input[["observed"]])
print(nrow(obs))
print(length(obs))
print(head(obs))

model <- snakemake@params[["model"]]
methylation <- snakemake@params[["methylation"]]

classes = c("Bile_Duct_Cancer", 
           "Breast_Cancer", 
           "Colorectal_Cancer", 
           "Gastric_cancer", 
           "Lung_Cancer", 
           "Ovarian_Cancer", 
           "Pancreatic_Cancer")

k_outer_cv = 10

combined_pred <- tibble(observed = rep(obs, k_outer_cv), 
                        CV_rep = rep(1:k_outer_cv, each=length(obs)))

combined_pred_PCA <- tibble(observed = rep(obs, k_outer_cv), 
                            CV_rep = rep(1:k_outer_cv, each=length(obs)))

for (type in 1:length(classes)){
    print(paste("Classification_output/", methylation, "Binomial_models_output/", model, "_Predictions_Full_data_", classes[type], ".rds", sep = ""))
    data_cancer_type <- readRDS(paste("Classification_output/", methylation, "Binomial_models_output/", model, "_Predictions_Full_data_", classes[type], ".rds", sep = ""))
    print(head(data_cancer_type))
    data_cancer_type <- data_cancer_type %>% select(colnames(data_cancer_type)[3])
    colnames(data_cancer_type) = paste(classes[type], "pred", sep = "_")
    print(head(data_cancer_type))
    combined_pred <- cbind(combined_pred, data_cancer_type)
    
    print(paste("Classification_output/", methylation, "Binomial_models_output/", model, "_Predictions_PCA_", classes[type], ".rds", sep = ""))
    data_cancer_type_PCA <- readRDS(paste("Classification_output/", methylation, "Binomial_models_output/", model, "_Predictions_PCA_", classes[type], ".rds", sep = ""))
    print(head(data_cancer_type_PCA))
    data_cancer_type_PCA <- data_cancer_type_PCA %>% select(colnames(data_cancer_type_PCA)[3])
    print(head(data_cancer_type_PCA))
    colnames(data_cancer_type_PCA) = paste(classes[type], "pred", sep = "_")
    combined_pred_PCA <- cbind(combined_pred_PCA, data_cancer_type_PCA)
}

IRdisplay::display(head(combined_pred))
saveRDS(combined_pred, file = snakemake@output[["cancer_type_predictions"]])

IRdisplay::display(head(combined_pred_PCA))
saveRDS(combined_pred_PCA, file = snakemake@output[["cancer_type_predictions_PCA"]])