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

k_outer_cv = 10

cal_auc <- function(X, Y) {
  id <- order(X)
  sum(diff(X[id])*rollmean(Y[id],2))
}

case_control_pred <- readRDS(snakemake@input[["predictions"]])
model <- snakemake@params[["model"]]

if (model == "Boosting"){
    model <- "GBM"
}

if (model == "lasso"){
    model <- "Elastic net"
}

if (model == "Lasso"){
    model <- "Elastic net"
}

head(case_control_pred)


predictions = list()
labels = list()
plot_list = list()
aucs_all <- tibble(cv_rep = seq(1:10))
class_aucs <- c()
class_FPRs_TPRs_avg <- tibble()

for (rep in 1:10){
    res_CV <- case_control_pred %>% filter(cv_rep == rep)
    print(head(res_CV))
    print(tail(res_CV))
    res_CV <- res_CV %>% select(-c(cv_rep)) 
    pred = as.vector(pull(res_CV, Cancer))
    predictions[[rep]] <- pred
    labels[[rep]] <- res_CV$observed

    pred_class_cv = prediction(pred, res_CV$observed,  label.ordering = c("Healthy", "Cancer"))
    perf_class_cv = performance(pred_class_cv, "tpr", "fpr")
    print("x.name")
    print(perf_class_cv@x.name)
    print("y.name")
    print(perf_class_cv@y.name)
    
    auc_class_cv <- performance(pred_class_cv, measure = "auc")
    class_aucs <- c(class_aucs, auc_class_cv@y.values[[1]])

    if (rep == 1){
        perf_class_cv_combined <- tibble(FPR = perf_class_cv@x.values[[1]], TPR = perf_class_cv@y.values[[1]], CV_rep = rep(rep, length(perf_class_cv@x.values[[1]])))
    } else {
        perf_class_cv_combined <- rbind(perf_class_cv_combined, tibble(FPR = perf_class_cv@x.values[[1]], TPR = perf_class_cv@y.values[[1]], CV_rep = rep(rep, length(perf_class_cv@x.values[[1]]))))
    }
}
aucs_all <- cbind(aucs_all, tibble(case_control_auc := class_aucs))

names(predictions) <- paste("CV_rep", seq(1:10), sep = "")
names(labels) <- paste("CV_rep", seq(1:10), sep = "")
pred = prediction(predictions, labels,  label.ordering = c("Healthy", "Cancer"))
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
perf_class_avg <- tibble(FPR = perf.avg@x.values[[1]], TPR = perf.avg@y.values[[1]])
class_FPRs_TPRs_avg <- rbind(class_FPRs_TPRs_avg, perf_class_avg)

aucs_class_avg <- tibble(case_control_auc := cal_auc(X=perf_class_avg$FPR, Y=perf_class_avg$TPR))

plot_list[[length(plot_list) + 1]] <- ggplot() + geom_line(data = perf_class_cv_combined, aes(x = FPR, y=TPR, group = CV_rep), color = "darkgrey", size = 0.5) + 
                                                 geom_line(data = perf_class_avg, aes(x = FPR, y=TPR), color = "firebrick", size = 1) + 
                                                 geom_abline(intercept = 0, slope = 1, color = "lightgrey", size = 0.5) +
                                                 theme_minimal() + ggtitle(gsub("_", " ", model, fixed=TRUE)) + 
                                                 theme(axis.text=element_text(size=20),
                                                       axis.title=element_text(size=20), 
                                                       plot.title = element_text(size=22))

#

aucs_class_avg <- aucs_class_avg %>% mutate(cv_rep = "mean") %>% select(cv_rep, everything())
aucs_all <- rbind(aucs_all, aucs_class_avg)

aucs_all
p <- plot_list[[1]]

saveRDS(aucs_all, snakemake@output[["aucs"]])

png(snakemake@output[["roc"]])
print(p)
dev.off()

####################################### Accuracy #######################################

eval_overall <- tibble()
accuracy_overall <- tibble()
class_eval <- c()
class_accuracy <- c()


for (rep in 1:k_outer_cv){
    res_cv_rep <- case_control_pred %>% filter(cv_rep == rep) %>% select(-c(cv_rep))

    pred_prob_class <- pull(res_cv_rep, Cancer)
    pred_prob_other <- 1 - pred_prob_class
    pred01 = as.factor(ifelse(pred_prob_class >= .5, "Cancer", "Healthy"))
    res <- tibble(obs = as.factor(res_cv_rep$observed), pred = pred01, Cancer = pred_prob_class, Healthy = pred_prob_other)
    cM_class_cvrep <- confusionMatrix(data = res$pred, reference = res$obs, positive = "Cancer")
    #print(cM_class_cvrep)

    class_eval <- c(cM_class_cvrep$byClass[["Balanced Accuracy"]], 
                    cM_class_cvrep$byClass[["Sensitivity"]], 
                    cM_class_cvrep$byClass[["Specificity"]], 
                    cM_class_cvrep$byClass[["Precision"]], 
                    cM_class_cvrep$byClass[["Recall"]])
    class_accuracy <- c(cM_class_cvrep$overall[["Accuracy"]], 
                        cM_class_cvrep$overall[["AccuracyLower"]], 
                        cM_class_cvrep$overall[["AccuracyUpper"]], 
                        cM_class_cvrep$overall[["AccuracyNull"]])
    
    eval_overall <- rbind(eval_overall, class_eval)
    accuracy_overall <- rbind(accuracy_overall, class_accuracy)
    }

colnames(eval_overall) <- c("Balanced Accuracy", "Sensitivity", "Specificity", "Precision", "Recall")
colnames(accuracy_overall) <- c("Accuracy", "AccuracyLower", "AccuracyUpper", "AccuracyNull")
eval_overall <- rbind(eval_overall, c(mean(eval_overall$`Balanced Accuracy`), 
                                      mean(eval_overall$Sensitivity),
                                      mean(eval_overall$Specificity),
                                      mean(eval_overall$Precision),
                                      mean(eval_overall$Recall)))

accuracy_overall <- rbind(accuracy_overall, c(mean(accuracy_overall$Accuracy), 
                                              mean(accuracy_overall$AccuracyLower),
                                              mean(accuracy_overall$AccuracyUpper),
                                              mean(accuracy_overall$AccuracyNull)))

eval_overall <- eval_overall %>% mutate(CV_rep = c(paste("CVrep", seq(1:10), sep = ""), "mean"))
accuracy_overall <- accuracy_overall %>% mutate(CV_rep = c(paste("CVrep", seq(1:10), sep = ""), "mean"))

saveRDS(eval_overall, snakemake@output[["eval_metrics"]])
saveRDS(accuracy_overall, snakemake@output[["accuracy"]])

print(eval_overall)
print(accuracy_overall)