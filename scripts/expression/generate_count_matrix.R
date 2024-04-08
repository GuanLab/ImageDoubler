library(tximport)

t2g <- read.table("../refs/transcript_to_gene.tsv", sep = "\t", header = F)

files <- list.files("../C1-SUM149-H1975/gene", pattern = ".h5", recursive = T, full.names = T)
filename <- unlist(lapply(files, function(x) strsplit(x, split = "/")[[1]][4]))
names(files) <- filename

txi.kallisto <- tximport(files, type = "kallisto", txOut = T)
matrix_img5_txlevel <- txi.kallisto$counts
write.table(matrix_img5_txlevel, file = "./counts/raw_counts_img5_txlevel.txt", sep = "\t", col.names = T, 
            row.names = T, quote = F)
txi.kallisto <- tximport(files, type = "kallisto", tx2gene = t2g, ignoreTxVersion = T)
matrix_img5_glevel <- txi.kallisto$counts
write.table(matrix_img5_glevel, file = "./counts/raw_counts_img5_glevel.txt", sep = "\t", col.names = T, 
            row.names = T, quote = F)


files <- list.files("../C1-SUM149-SUM190/gene", pattern = ".h5", recursive = T, full.names = T)
filename <- unlist(lapply(files, function(x) strsplit(x, split = "/")[[1]][4]))
names(files) <- filename

txi.kallisto <- tximport(files, type = "kallisto", txOut = T)
matrix_img11_txlevel <- txi.kallisto$counts
write.table(matrix_img11_txlevel, file = "./counts/raw_counts_img11_txlevel.txt", sep = "\t", col.names = T, 
            row.names = T, quote = F)
txi.kallisto <- tximport(files, type = "kallisto", tx2gene = t2g, ignoreTxVersion = T)
matrix_img11_glevel <- txi.kallisto$counts
write.table(matrix_img11_glevel, file = "./counts/raw_counts_img11_glevel.txt", sep = "\t", col.names = T, 
            row.names = T, quote = F)


