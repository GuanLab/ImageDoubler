
library("DoubletFinder")
library(Seurat)
library(reticulate)

rm(list = ls())
setwd("~/Desktop/GuanLab/single-cell-image/expression/DoubletFinder/")

# without gold standards
counts_img5_glevel <- read.table("../raw_counts_img5_glevel.txt", sep = "\t", header = T)
counts_img11_glevel <- read.table("../raw_counts_img11_glevel.txt", sep = "\t", header = T)
targets <- read.csv("../targets.csv")

run_doubletfinder <- function(count_matrix, image_set, include_only_singlet_doublet = F) {
    targets <- targets[targets$image_set == image_set, ]
    targets$class <- stringr::str_to_title(targets$class)
    if (include_only_singlet_doublet) {
        remove_imgs = targets[targets$class == "Missing", "image_pos"]
        count_matrix <- dplyr::select(count_matrix, -remove_imgs)
    }
    
    seu <- CreateSeuratObject(count_matrix)
    seu <- NormalizeData(seu)
    seu <- FindVariableFeatures(seu)
    seu <- ScaleData(seu)
    seu <- RunPCA(seu)
    seu <- FindNeighbors(seu, dims = 1:20)
    seu <- FindClusters(seu)
    seu <- RunUMAP(seu, dims = 1:20)
    
    sweep.res.seu <- paramSweep(seu, PCs = 1:20, sct = FALSE)
    sweep.seu <- summarizeSweep(sweep.res.seu, GT = FALSE)
    bcmvn <- find.pK(sweep.seu)
    pK <- as.numeric(as.character(bcmvn[bcmvn$BCmetric == max(bcmvn$BCmetric), "pK"]))
    
    annotations <- seu@meta.data$seurat_clusters
    homotypic.prop <- modelHomotypic(annotations)           ## ex: annotations <- seu_kidney@meta.data$ClusteringResults
    nExp_poi <- round(0.05*nrow(seu@meta.data))  
    nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
    
    seu <- doubletFinder(seu, 
                         PCs = 1:20, 
                         pN = 0.25, 
                         pK = pK, 
                         nExp = nExp_poi, 
                         reuse.pANN = FALSE, 
                         sct = FALSE)
    
    meta.data <- seu@meta.data
    meta.data$image_pos <- row.names(meta.data)
    meta.data <- merge(meta.data, targets, by = "image_pos", all.x = T)
    return(list(seu = seu, meta = meta.data))
}

evaluate <- function(meta.data) {
    sklearn <- import("sklearn.metrics")
    np <- import("numpy")
    meta.data <- meta.data[meta.data$class %in% c("Singlet", "Doublet"), ]
    accuracy <- sklearn$accuracy_score(meta.data[, grep("DF", colnames(meta.data))], meta.data$class)
    f1 <- sklearn$f1_score(meta.data[, grep("DF", colnames(meta.data))], meta.data$class, pos_label="Doublet")
    confusion <- sklearn$confusion_matrix(meta.data[, grep("DF", colnames(meta.data))], meta.data$class, 
                                          labels=c("Singlet", "Doublet"))
    # accuracy, f1, tn, fn, fp, tp
    results <- c(accuracy, f1, np$ravel(confusion))
    return(results)
}

seu_img5 <- run_doubletfinder(counts_img5_glevel, "Image5", include_only_singlet_doublet = F)
seu_img5_noMissing <- run_doubletfinder(counts_img5_glevel, "Image5", include_only_singlet_doublet = T)

write.csv(seu_img5$meta, "results_img5.csv", row.names = F)
write.csv(seu_img5_noMissing$meta, "results_img5_noMissing.csv", row.names = F)

res_img5 <- evaluate(seu_img5$meta)
res_img5_noMissing <- evaluate(seu_img5_noMissing$meta)

seu_img11 <- run_doubletfinder(counts_img11_glevel, "Image11", include_only_singlet_doublet = F)
seu_img11_noMissing <- run_doubletfinder(counts_img11_glevel, "Image11", include_only_singlet_doublet = T)

write.csv(seu_img11$meta, "results_img11.csv", row.names = F)
write.csv(seu_img11_noMissing$meta, "results_img11_noMissing.csv", row.names = F)

res_img11 <- evaluate(seu_img11$meta)
res_img11_noMissing <- evaluate(seu_img11_noMissing$meta)

scores <- as.data.frame(
    rbind(res_img5, res_img5_noMissing, 
          res_img11, res_img11_noMissing)
)
colnames(scores) <- c("accuracy", "f1", "tn", "fn", "fp", "tp")
scores$method <- "DoubletFinder"
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
scores$method <- "DoubletFinder"
write.csv(scores, "scores.csv")
