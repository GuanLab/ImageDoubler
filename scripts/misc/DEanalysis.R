
library(DESeq2)
library(tximport)
library(dplyr)
library(stringr)

# receive outside arguments
args <- commandArgs(trailingOnly = TRUE)

# args pass to detection_method, method_name
detection_method <- args[1]
method_name <- args[2]

t2g <- read.table("expression/refs/transcript_to_gene.tsv", sep = "\t", header = F)

get_de_img5 <- function(obs_data, detection_method, method_name, save_dir) {
    files <- list.files("expression/C1-SUM149-H1975/gene", pattern = ".h5", recursive = T, full.names = T)
    filename <- unlist(lapply(files, function(x) strsplit(x, split = "/")[[1]][4]))
    names(files) <- filename

    # obs_img5 <- read.csv("Figure5/umap_data_removal_img5.csv")
    obs_img5 <- read.csv(obs_data)
    obs_img5 <- obs_img5[obs_img5$detectiont_method == detection_method, ]
    row.names(obs_img5) <- obs_img5$image_pos
    obs_img5$leiden <- as.character(obs_img5$leiden)
    files <- files[names(files) %in% obs_img5$image_pos]
    txi.kallisto <- tximport(files, type = "kallisto", tx2gene = t2g, ignoreTxVersion = T)

    dds <- DESeqDataSetFromTximport(txi.kallisto, obs_img5, ~cell_type)
    dds <- DESeq(dds)
    res <- results(dds)
    # res_df <- as.data.frame(res) %>% filter(!is.na(padj) & (padj < 0.005) & abs(log2FoldChange) >= 1)
    res_df <- as.data.frame(res) %>% filter(!is.na(padj))
    write.csv(res_df, file=str_interp("Figure5/${save_dir}/DEGs_celltype_${method_name}_img5.csv"))

    dds_leiden <- DESeqDataSetFromTximport(txi.kallisto, obs_img5, ~leiden)
    dds_leiden <- DESeq(dds_leiden)

    if (length(unique(obs_img5$leiden)) == 3) {
        res_df <- rbind(
            as.data.frame(results(dds_leiden, contrast=c("leiden", "0", "1"))) %>% 
                filter(!is.na(padj)) %>%
                mutate(comparison = "leiden_0_vs_1"),
            as.data.frame(results(dds_leiden, contrast=c("leiden", "0", "2"))) %>% 
                filter(!is.na(padj)) %>%
                mutate(comparison = "leiden_0_vs_2"),
            as.data.frame(results(dds_leiden, contrast=c("leiden", "1", "2"))) %>% 
                filter(!is.na(padj)) %>%
                mutate(comparison = "leiden_1_vs_2")
        )
    } else {
        res_df <- as.data.frame(results(dds_leiden, contrast=c("leiden", "0", "1"))) %>% 
            filter(!is.na(padj)) %>%
            mutate(comparison = "leiden_0_vs_1")
    }
    
    write.csv(res_df, file=str_interp("Figure5/${save_dir}/DEGs_leiden_${method_name}_img5.csv"))
}


get_de_img11 <- function(obs_data, detection_method, method_name, save_dir) {
    files <- list.files("expression/C1-SUM149-SUM190/gene", pattern = ".h5", recursive = T, full.names = T)
    filename <- unlist(lapply(files, function(x) strsplit(x, split = "/")[[1]][4]))
    names(files) <- filename

    # obs_img5 <- read.csv("Figure5/umap_data_removal_img11.csv")
    obs_img5 <- read.csv(obs_data)
    obs_img5 <- obs_img5[obs_img5$detectiont_method == detection_method, ]
    row.names(obs_img5) <- obs_img5$image_pos
    obs_img5$leiden <- as.character(obs_img5$leiden)
    files <- files[names(files) %in% obs_img5$image_pos]
    txi.kallisto <- tximport(files, type = "kallisto", tx2gene = t2g, ignoreTxVersion = T)

    dds <- DESeqDataSetFromTximport(txi.kallisto, obs_img5, ~cell_type)
    dds <- DESeq(dds)
    res <- results(dds)
    res_df <- as.data.frame(res) %>% filter(!is.na(padj))
    write.csv(res_df, file=str_interp("Figure5/${save_dir}/DEGs_celltype_${method_name}_img11.csv"))

    dds_leiden <- DESeqDataSetFromTximport(txi.kallisto, obs_img5, ~leiden)
    dds_leiden <- DESeq(dds_leiden)
    
    if (length(unique(obs_img5$leiden)) == 3) {
        res_df <- rbind(
            as.data.frame(results(dds_leiden, contrast=c("leiden", "0", "1"))) %>% 
                filter(!is.na(padj)) %>%
                mutate(comparison = "leiden_0_vs_1"),
            as.data.frame(results(dds_leiden, contrast=c("leiden", "0", "2"))) %>% 
                filter(!is.na(padj)) %>%
                mutate(comparison = "leiden_0_vs_2"),
            as.data.frame(results(dds_leiden, contrast=c("leiden", "1", "2"))) %>% 
                filter(!is.na(padj)) %>%
                mutate(comparison = "leiden_1_vs_2")
        )
    } else {
        res_df <- as.data.frame(results(dds_leiden, contrast=c("leiden", "0", "1"))) %>% 
            filter(!is.na(padj)) %>%
            mutate(comparison = "leiden_0_vs_1")
    }

    write.csv(res_df, file=str_interp("Figure5/${save_dir}/DEGs_leiden_${method_name}_img11.csv"))
}

get_de_img5("Figure5/umap_data_removal_img5.csv", detection_method, method_name, "DEGs")
get_de_img11("Figure5/umap_data_removal_img11.csv", detection_method, method_name, "DEGs")
get_de_img5("Figure5/umap_data_removal_img5_noMissing.csv", detection_method, method_name, "DEGs_noMissing")
get_de_img11("Figure5/umap_data_removal_img11_noMissing.csv", detection_method, method_name, "DEGs_noMissing")

# get_de_img5("Figure5/umap_data_removal_img5.csv", "no removal", "noRemoval", "DEGs")
# get_de_img5("Figure5/umap_data_removal_img5.csv", "Ground-truth", "HandLabel", "DEGs")
# get_de_img5("Figure5/umap_data_removal_img5.csv", "ImageDoubler", "ImageDoubler", "DEGs")

# get_de_img11("Figure5/umap_data_removal_img11.csv", "no removal", "noRemoval", "DEGs")
# get_de_img11("Figure5/umap_data_removal_img11.csv", "Ground-truth", "HandLabel", "DEGs")
# get_de_img11("Figure5/umap_data_removal_img11.csv", "ImageDoubler", "ImageDoubler", "DEGs")


# dds_results[["img5_dd"]] <- get_de_img5("DoubletDetection", "DoubletDetection")
# dds_results[["img5_scrub"]] <- get_de_img5("Scrublet", "Scrublet")
# dds_results[["img5_solo"]] <- get_de_img5("Solo", "Solo")
# dds_results[["img5_scds"]] <- get_de_img5("scds", "scds")
# dds_results[["img5_scdbl"]] <- get_de_img5("scDblFinder", "scDblFinder")
# dds_results[["img5_df"]] <- get_de_img5("DoubletFinder", "DoubletFinder")

# dds_results[["img11_dd"]] <- get_de_img11("DoubletDetection", "DoubletDetection")
# dds_results[["img11_scrub"]] <- get_de_img11("Scrublet", "Scrublet")
# dds_results[["img11_solo"]] <- get_de_img11("Solo", "Solo")
# dds_results[["img11_scds"]] <- get_de_img11("scds", "scds")
# dds_results[["img11_scdbl"]] <- get_de_img11("scDblFinder", "scDblFinder")
# dds_results[["img11_df"]] <- get_de_img11("DoubletFinder", "DoubletFinder")
