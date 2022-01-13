library(rlist)
library(tidyverse)
library(multiROC)
library(dummies)
library(ROCR)
install.packages("cowplot", repos = "http://cran.us.r-project.org/src/contrib/cowplot_1.1.1.tar.gz")
library(cowplot)
library(zoo)
install.packages("LICORS", repos = "http://cran.us.r-project.org/src/contrib/LICORS_0.2.0.tar.gz")
library(LICORS)
library(caret)

cal_auc <- function(X, Y) {
  id <- order(X)
  sum(diff(X[id])*rollmean(Y[id],2))
}

classes = c("Bile_Duct_Cancer", 
           "Breast_Cancer", 
           "Colorectal_Cancer", 
           "Gastric_cancer", 
           "Lung_Cancer", 
           "Ovarian_Cancer", 
           "Pancreatic_Cancer")
k_outer_cv = 10

classification_type <- snakemake@params[["classification_type"]] 
input_predictions <- readRDS(snakemake@input[["predictions"]])
observed_vec <- readRDS(snakemake@input[["observed"]])

model <- snakemake@params[["model"]]

if (model == "Boosting"){
    model <- "GBM"
}

if (model == "Lasso"){
    model <- "Elastic net"
}

if (model == "SVM_poly"){
    model <- "SVM poly"
}

if (model == "SVM_radial"){
    model <- "SVM radial"
}

if (model == "SVM_linear"){
    model <- "SVM linear"
}

colnames(input_predictions) <- sub("_pr.*", "", colnames(input_predictions))

print("The head of predictions")
head(input_predictions)

print("The head of observed")
head(observed_vec)

predictions = list()
labels = list()
plot_list = list()
aucs_all <- tibble(cv_rep = seq(1:10))
class_FPRs_TPRs_avg <- tibble()

for (type in 1:length(classes)) {
    predictions = list()
    labels = list()
    myplot <- NULL
    class <- paste0(classes[type], "")
    obs <- ifelse(observed_vec == class, 1, 0)
    print(class)
    class_aucs <- c()
    
    for (CV_REP in 1:k_outer_cv){
        res_CV <- input_predictions %>% filter(cv_rep == CV_REP) %>% select(-c(observed, cv_rep)) 
        pred = as.vector(pull(res_CV, class))
        predictions[[CV_REP]] <- pred
        labels[[CV_REP]] <- obs
       
        pred_class_cv = prediction(pred, obs)
        perf_class_cv = performance(pred_class_cv, "tpr", "fpr")
        
        auc_class_cv <- performance(pred_class_cv, measure = "auc")
        class_aucs <- c(class_aucs, auc_class_cv@y.values[[1]])
        
        if (CV_REP == 1){
            perf_class_cv_combined <- tibble(FPR = perf_class_cv@x.values[[1]], TPR = perf_class_cv@y.values[[1]], CV_rep = rep(CV_REP, length(perf_class_cv@x.values[[1]])))
        } else {
            perf_class_cv_combined <- rbind(perf_class_cv_combined, tibble(FPR = perf_class_cv@x.values[[1]], TPR = perf_class_cv@y.values[[1]], CV_rep = rep(CV_REP, length(perf_class_cv@x.values[[1]]))))
        }
    }
    aucs_all <- cbind(aucs_all, tibble("{class}_" := class_aucs))
    
    names(predictions) <- paste("CV_rep", seq(1:10), sep = "")
    names(labels) <- paste("CV_rep", seq(1:10), sep = "")
    pred = prediction(predictions, labels)
    perf = performance(pred, "tpr", "fpr") # x.name FPR, y.name TPR
    
    ##############################################################################
    if (length(perf@alpha.values) != 0) {
      FUN <- function(x) {
        isfin <- is.finite(x)
        # if only one finite is available the mean cannot be calculated without
        # the first/last value, since the leaves no value
        if(sum(isfin) > 1L){ 
          inf_replace <- max(x[isfin]) + 
            mean(abs(x[isfin][-1] - x[isfin][-length(x[isfin])]))
        } else {
          inf_replace <- 0
        }
        x[is.infinite(x)] <- inf_replace
        x
      }
      perf@alpha.values <- lapply(perf@alpha.values,FUN)
    }

    ##############################################################################
    
    perf.sampled <- perf
    alpha.values <- rev(seq(min(unlist(perf@alpha.values)),
                          max(unlist(perf@alpha.values)),
                          length=max( sapply(perf@alpha.values, length))))
    
    for (i in 1:length(perf.sampled@y.values)) {
        perf.sampled@x.values[[i]] <- stats::approxfun(perf@alpha.values[[i]],perf@x.values[[i]],
                       rule=2, ties=mean)(alpha.values)
        perf.sampled@y.values[[i]] <- stats::approxfun(perf@alpha.values[[i]], perf@y.values[[i]],
                       rule=2, ties=mean)(alpha.values)
    }

    ## compute average curve
    perf.avg <- perf.sampled
    perf.avg@x.values <- list(rowMeans( data.frame( perf.avg@x.values)))
    perf.avg@y.values <- list(rowMeans( data.frame( perf.avg@y.values)))
    perf.avg@alpha.values <- list( alpha.values )
    perf_class_avg <- tibble(FPR = perf.avg@x.values[[1]], TPR = perf.avg@y.values[[1]], class_type = rep(class, length(perf.avg@y.values[[1]])))
    class_FPRs_TPRs_avg <- rbind(class_FPRs_TPRs_avg, perf_class_avg)
    
    if (type == 1){
        aucs_class_avg <- tibble("{class}_" := cal_auc(X=perf_class_avg$FPR, Y=perf_class_avg$TPR))
    } else {
        aucs_class_avg <- cbind(aucs_class_avg, tibble("{class}_" := cal_auc(X=perf_class_avg$FPR, Y=perf_class_avg$TPR)))
    }
    plot_list[[length(plot_list) + 1]] <- ggplot() + geom_line(data = perf_class_cv_combined, aes(x = FPR, y=TPR, group = CV_rep), color = "darkgrey", size = 0.5) + 
                                                     geom_line(data = perf_class_avg, aes(x = FPR, y=TPR), color = "firebrick", size = 1) + 
                                                     geom_abline(intercept = 0, slope = 1, color = "lightgrey", size = 0.5) +
                                                     theme_minimal() + 
                                                     ggtitle(gsub("_", " ", class, fixed=TRUE)) + 
                                                     theme(axis.text=element_text(size=10),
                                                           axis.title=element_text(size=12), 
                                                           plot.title = element_text(size=12))
    
}

aucs_class_avg <- aucs_class_avg %>% mutate(cv_rep = "mean") %>% select(cv_rep, everything())
aucs_all <- rbind(aucs_all, aucs_class_avg)
print("AUCs")
print(aucs_all)
saveRDS(aucs_all, snakemake@output[["aucs"]])
    
p <- plot_grid(plot_list[[1]], plot_list[[2]], plot_list[[3]], plot_list[[4]], plot_list[[5]], plot_list[[6]], plot_list[[7]], nrow = 3)
p1 <- ggplot(class_FPRs_TPRs_avg) + geom_line(aes(x = FPR, y = TPR, color = class_type), size = 1) + theme_minimal() + 
        ggtitle(model) +
        geom_abline(intercept = 0, slope = 1, color = "lightgrey", size = 0.5) +
                                                 theme(axis.text=element_text(size=22),
                                                       axis.title=element_text(size=22), 
                                                       plot.title = element_text(size=26), 
                                                       legend.position = "none")


png(snakemake@output[["roc_grid"]])
print(p)
dev.off()

png(snakemake@output[["roc_avg"]])
print(p1)
dev.off()


if (classification_type == "binomial"){
    
    eval_overall <- tibble(Class = character(), 
                           `Balanced Accuracy` = numeric(), 
                           Sensitivity = numeric(), 
                           Specificity = numeric(), 
                           Precision = numeric(), 
                           Recall = numeric() 
                          )
    accuracy_overall <- tibble(Class = character(), 
                               Accuracy = numeric(),
                               AccuracyLower = numeric(),
                               AccuracyUpper = numeric(),
                               AccuracyNull = numeric() 
                              )
    
    for (type in 1:length(classes)) {
        class <- paste0(classes[type], "")
        eval_class <- tibble()
        accuracy_class <- tibble()

        for (CV_REP in 1:k_outer_cv){
            res_cv_rep <- input_predictions %>% filter(cv_rep == CV_REP) %>% select(-c(cv_rep))
            
            obs <- as.factor(ifelse(res_cv_rep$observed == class, class, "Other"))
            #print(levels(obs))
            pred_prob_class <- pull(res_cv_rep, class)
            pred_prob_other <- 1 - pred_prob_class
            pred01 = as.factor(ifelse(pred_prob_class >= .5, class, "Other"))
            res <- tibble(obs = obs, pred = pred01, "{class}" := pred_prob_class, Other = pred_prob_other)
            cM_class_cvrep <- confusionMatrix(data = res$pred, reference = res$obs, positive = class)
            #print(cM_class_cvrep)
            
            class_cvrep_eval <- tibble(class = class, 
                                       Sensitivity = cM_class_cvrep$byClass[["Sensitivity"]], 
                                       Specificity = cM_class_cvrep$byClass[["Specificity"]], 
                                       `Balanced Accuracy` = cM_class_cvrep$byClass[["Balanced Accuracy"]], 
                                       Precision = cM_class_cvrep$byClass[["Precision"]], 
                                       Recall = cM_class_cvrep$byClass[["Recall"]])
            class_cvrep_acc <- tibble(class = class, 
                                      Accuracy = cM_class_cvrep$overall[["Accuracy"]], 
                                      AccuracyLower = cM_class_cvrep$overall[["AccuracyLower"]], 
                                      AccuracyUpper = cM_class_cvrep$overall[["AccuracyUpper"]], 
                                      AccuracyNull = cM_class_cvrep$overall[["AccuracyNull"]])
            
        eval_class <- rbind(eval_class, class_cvrep_eval)
        accuracy_class <- rbind(accuracy_class, class_cvrep_acc)
        
        }
        
        
        eval_class <- rbind(eval_class, c(class, 
                                          mean(eval_class$`Balanced Accuracy`), 
                                          mean(eval_class$Sensitivity),
                                          mean(eval_class$Specificity),
                                          mean(eval_class$Precision),
                                          mean(eval_class$Recall)))

        accuracy_class <- rbind(accuracy_class, c(class, 
                                                  mean(accuracy_class$Accuracy), 
                                                  mean(accuracy_class$AccuracyLower),
                                                  mean(accuracy_class$AccuracyUpper),
                                                  mean(accuracy_class$AccuracyNull)))

        eval_class <- eval_class %>% mutate(CV_rep = c(paste("CVrep", seq(1:10), sep = ""), "mean"))
        accuracy_class <- accuracy_class %>% mutate(CV_rep = c(paste("CVrep", seq(1:10), sep = ""), "mean"))
        
        eval_overall <- rbind(eval_overall, eval_class)
        accuracy_overall <- rbind(accuracy_overall, accuracy_class)
        
    }
    saveRDS(eval_overall, snakemake@output[["eval_metrics"]])
    saveRDS(accuracy_overall, snakemake@output[["accuracy"]])
    
    print("eval_overall")
    print(eval_overall)
    print("accuracy_overall")
    print(accuracy_overall)

}


if (classification_type == "multinomial"){
    accuracy <- c()
    acc_lower <- c()
    acc_upper <- c()
    acc_null <- c()
    class_res <- tibble()
    for (CV_REP in 1:k_outer_cv){
        res_cv_rep <- input_predictions %>% filter(cv_rep == CV_REP) %>% select(-c(cv_rep))
        #IRdisplay::display(head(res_cv_rep))
        res_cv_rep_probs <- res_cv_rep %>% select(-c(observed))
        res_cv_rep_probs <- res_cv_rep_probs %>% mutate(pred01 = factor(colnames(res_cv_rep_probs)[apply(res_cv_rep_probs,1,which.max)], ordered = TRUE))
        res <- tibble(obs = factor(res_cv_rep$observed), pred = factor(res_cv_rep_probs$pred01, levels = levels(factor(res_cv_rep$observed))))
        cM_cvrep <- confusionMatrix(data = res$pred, reference = res$obs)
        #print(cM_cvrep)
        class_names <- rownames(cM_cvrep$byClass)
        res_class_cv <- as_tibble(cM_cvrep$byClass[1:7, c("Sensitivity", "Specificity", "Precision", "Recall", "Balanced Accuracy")])
        res_class_cv <- res_class_cv %>% mutate(class_name = class_names, CV_rep = rep(CV_REP, nrow(res_class_cv)))
        accuracy <- c(accuracy, cM_cvrep$overall[["Accuracy"]])
        acc_lower <- c(acc_lower, cM_cvrep$overall[["AccuracyLower"]])
        acc_upper <- c(acc_upper, cM_cvrep$overall[["AccuracyUpper"]])
        acc_null <- c(acc_null, cM_cvrep$overall[["AccuracyNull"]])
        class_res <- rbind(class_res, res_class_cv)
    }
    accuracy <- c(accuracy, mean(accuracy))
    acc_lower <- c(acc_lower, mean(acc_lower))
    acc_upper <- c(acc_upper, mean(acc_upper))
    acc_null <- c(acc_null, mean(acc_null))
    
    mean_res_class <- class_res %>% group_by(class_name) %>% summarise(Sensitivity = mean(Sensitivity), 
                                                                       Specificity = mean(Specificity),
                                                                       Precision = mean(Precision),
                                                                       Recall = mean(Recall),
                                                                       `Balanced Accuracy` = mean(`Balanced Accuracy`)
                                                                      ) %>% mutate(CV_rep = rep("mean", 7))
    
    accuracy_overall <- tibble("Accuracy" := accuracy, 
                               "AccuracyLower" := acc_lower,
                               "AccuracyUpper" := acc_upper,
                               "AccuracyNull" := acc_null) %>% mutate(CV_rep = c(paste("CVrep", seq(1:k_outer_cv), sep = ""), "mean"))

    res_class <- rbind(class_res, mean_res_class)

    print("Sensitivity, specificity, precision, recall, balanced accuracy, per class")    
    print(res_class)
    print("Overall accuracy, per class")  
    print(accuracy_overall)
    
    saveRDS(res_class, snakemake@output[["eval_metrics"]])
    saveRDS(accuracy_overall, snakemake@output[["accuracy"]])
    
}


