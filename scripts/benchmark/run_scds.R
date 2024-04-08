
library(SingleCellExperiment)
library("scds")
library(reticulate)


setwd("~/Desktop/GuanLab/single-cell-image/expression/scds/")

counts_img5_glevel <- read.table("../raw_counts_img5_glevel.txt", sep = "\t", header = T)
counts_img11_glevel <- read.table("../raw_counts_img11_glevel.txt", sep = "\t", header = T)
targets <- read.csv("../targets.csv")

min_max_norm <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
}

run_scds <- function(df_counts, image_set, include_only_singlet_doublet = F) {
    targets <- targets[targets$image_set == image_set, ]
    if (include_only_singlet_doublet) {
        remove_imgs = targets[targets$class == "missing", "image_pos"]
        df_counts <- dplyr::select(df_counts, -remove_imgs)
    }
    
    sce <- SingleCellExperiment(list(counts=as.matrix(df_counts)))
    sce <- cxds(sce)
    sce <- bcds(sce)
    sce <- cxds_bcds_hybrid(sce)
    
    doublet_scores <- as.data.frame(colData(sce))
    doublet_scores$cxds_score <- min_max_norm(doublet_scores$cxds_score)
    doublet_scores$bcds_score <- min_max_norm(doublet_scores$bcds_score)
    doublet_scores$hybrid_score <- min_max_norm(doublet_scores$hybrid_score)
    
    doublet_scores$image_pos <- row.names(doublet_scores)
    doublet_scores <- merge(doublet_scores, targets, by = "image_pos", all.x = T)
    return(doublet_scores)
}

evaluate <- function(meta.data, thresh, method) {
    sklearn <- import("sklearn.metrics")
    np <- import("numpy")
    meta.data <- meta.data[meta.data$class %in% c("singlet", "doublet"), ]
    meta.data$preds <- ifelse(meta.data[, method] > thresh, "doublet", "singlet")
    accuracy <- sklearn$accuracy_score(meta.data$preds, meta.data$class)
    f1 <- sklearn$f1_score(meta.data$preds, meta.data$class, pos_label="doublet")
    confusion <- sklearn$confusion_matrix(meta.data$preds, meta.data$class, labels=c("singlet", "doublet"))
    # accuracy, f1, tn, fn, fp, tp, thresh, method
    results <- c(accuracy, f1, np$ravel(confusion), thresh, method)
    return(results)
}

img5_doublet_scores <- run_scds(counts_img5_glevel, "Image5")
img5_doublet_scores_noMissing <- run_scds(counts_img5_glevel, "Image5", include_only_singlet_doublet = T)
img11_doublet_scores <- run_scds(counts_img11_glevel, "Image11")
img11_doublet_scores_noMissing <- run_scds(counts_img11_glevel, "Image11", include_only_singlet_doublet = T)

scores <- c()
for (method in c("cxds_score", "bcds_score", "hybrid_score")) {
    for (thresh in c(0.3, 0.4, 0.5, 0.6, 0.7)) {
        img5_eva <- evaluate(img5_doublet_scores, thresh, method)
        img5_eva_noMissing <- evaluate(img5_doublet_scores_noMissing, thresh, method)
        img11_eva <- evaluate(img11_doublet_scores, thresh, method)
        img11_eval_noMissing <- evaluate(img11_doublet_scores_noMissing, thresh, method)
        scores <- rbind(scores, img5_eva, img5_eva_noMissing, img11_eva, img11_eval_noMissing)
    }
}
scores <- as.data.frame(scores)
colnames(scores) <- c("accuracy", "f1", "tn", "fn", "fp", "tp", "thresh", "method")
write.csv(scores, "scores.csv")

write.csv(img5_doublet_scores, file = "scds_img5_doublet_scores.csv")
write.csv(img5_doublet_scores_noMissing, file = "scds_img5_doublet_scores_noMissing.csv")
write.csv(img11_doublet_scores, file = "scds_img11_doublet_scores.csv")
write.csv(img11_doublet_scores_noMissing, file = "scds_img11_doublet_scores_noMissing.csv")

# re evaluate to remove the difficult
img5_doublet_scores <- read.csv("./scds_img5_doublet_scores.csv") %>% 
    filter(difficult == "False")
img5_doublet_scores_noMissing <- read.csv("./scds_img5_doublet_scores_noMissing.csv") %>% 
    filter(difficult == "False")
img11_doublet_scores <- read.csv("./scds_img11_doublet_scores.csv") %>% 
    filter(difficult == "False")
img11_doublet_scores_noMissing <- read.csv("./scds_img11_doublet_scores_noMissing.csv") %>% 
    filter(difficult == "False")

scores <- c()
for (method in c("cxds_score", "bcds_score", "hybrid_score")) {
    for (thresh in c(0.3, 0.4, 0.5, 0.6, 0.7)) {
        img5_eva <- evaluate(img5_doublet_scores, thresh, method)
        img5_eva_noMissing <- evaluate(img5_doublet_scores_noMissing, thresh, method)
        img11_eva <- evaluate(img11_doublet_scores, thresh, method)
        img11_eval_noMissing <- evaluate(img11_doublet_scores_noMissing, thresh, method)
        scores <- rbind(scores, img5_eva, img5_eva_noMissing, img11_eva, img11_eval_noMissing)
    }
}
scores <- as.data.frame(scores)
colnames(scores) <- c("accuracy", "f1", "tn", "fn", "fp", "tp", "thresh", "method")
write.csv(scores, "scores.csv")
