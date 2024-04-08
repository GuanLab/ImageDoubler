
library(SingleCellExperiment)
library(scDblFinder)
library(reticulate)

rm(list = ls())
setwd("~/Desktop/GuanLab/single-cell-image/expression/scDblFinder/")

counts_img5_glevel <- read.table("../raw_counts_img5_glevel.txt", sep = "\t", header = T)
counts_img11_glevel <- read.table("../raw_counts_img11_glevel.txt", sep = "\t", header = T)
targets <- read.csv("../targets.csv")

run_finder <- function(count_matrix, image_set, include_only_singlet_doublet = F, nfeatures = NA) {
    targets <- targets[targets$image_set == image_set, ]
    if (include_only_singlet_doublet) {
        remove_imgs = targets[targets$class == "missing", "image_pos"]
        count_matrix <- dplyr::select(count_matrix, -remove_imgs)
    }
    
    sce <- SingleCellExperiment(list(counts=as.matrix(count_matrix)))
    if (!is.na(nfeatures)) {
        sce <- scDblFinder(sce, nfeatures = nfeatures)
    } else {
        sce <- scDblFinder(sce)
    }
    
    preds <- as.data.frame(colData(sce))
    preds$image_pos = row.names(preds)
    preds <- merge(preds, targets, by = "image_pos", all.x = T)
    return(preds)
}

evaluate <- function(meta.data) {
    sklearn <- import("sklearn.metrics")
    np <- import("numpy")
    meta.data <- meta.data[meta.data$class %in% c("singlet", "doublet"), ]
    accuracy <- sklearn$accuracy_score(meta.data[, "scDblFinder.class"], meta.data$class)
    f1 <- sklearn$f1_score(meta.data[, "scDblFinder.class"], meta.data$class, pos_label="doublet")
    confusion <- sklearn$confusion_matrix(meta.data[, "scDblFinder.class"], meta.data$class, labels=c("singlet", "doublet"))
    # accuracy, f1, tn, fn, fp, tp
    results <- c(accuracy, f1, np$ravel(confusion))
    return(results)
}

res_img5 <- run_finder(counts_img5_glevel, "Image5", nfeatures = 3000)
write.csv(res_img5, "results_img5.csv", row.names = F)
res_img5_noMissing <- run_finder(counts_img5_glevel, "Image5", T, nfeatures = 3000)
write.csv(res_img5_noMissing, "results_img5_noMissing.csv", row.names = F)

score_img5 <- evaluate(res_img5)
score_img5_noMissing <- evaluate(res_img5_noMissing)

res_img11 <- run_finder(counts_img11_glevel, "Image11", nfeatures = 1000)
write.csv(res_img11, "results_img11.csv", row.names = F)
res_img11_noMissing <- run_finder(counts_img11_glevel, "Image11", T, nfeatures = 1000)
write.csv(res_img11_noMissing, "results_img11_noMissing.csv", row.names = F)

score_img11 <- evaluate(res_img11)
score_img11_noMissing <- evaluate(res_img11_noMissing)

scores <- as.data.frame(rbind(score_img5, score_img5_noMissing, score_img11, score_img11_noMissing))
colnames(scores) <- c("accuracy", "f1", "tn", "fn", "fp", "tp")
scores$method <- "scDblFinder"
write.csv(scores, "scores.csv")

# re evaluate
res_img5 <- read.csv("results_img5.csv") %>% 
    filter(difficult == "False")
res_img5_noMissing <- read.csv("results_img5_noMissing.csv") %>% 
    filter(difficult == "False")
res_img11 <- read.csv("results_img11.csv") %>% 
    filter(difficult == "False")
res_img11_noMissing <- read.csv("results_img11_noMissing.csv") %>% 
    filter(difficult == "False")

score_img5 <- evaluate(res_img5)
score_img5_noMissing <- evaluate(res_img5_noMissing)
score_img11 <- evaluate(res_img11)
score_img11_noMissing <- evaluate(res_img11_noMissing)

scores <- as.data.frame(rbind(score_img5, score_img5_noMissing, score_img11, score_img11_noMissing))
colnames(scores) <- c("accuracy", "f1", "tn", "fn", "fp", "tp")
scores$method <- "scDblFinder"
write.csv(scores, "scores.csv")
