suppressPackageStartupMessages({
  library(argparser)
  library(rtracklayer)
  library(MatrixGenerics)
  library(GenomicRanges)
})

p <- arg_parser("Annotate H3K9ac domains with count stats + Roadmap chromatin-state overlap and cluster them")
p <- add_argument(p, "--counts-rds",   help = "H3K9acCounts.rds (from 03_generateMatrix.R)")
p <- add_argument(p, "--qc",           help = "qualityMetrics.csv")
p <- add_argument(p, "--sample-sheet", help = "sampleSheet.csv")
p <- add_argument(p, "--lib",          help = "Path to readH3k9acCounts.R")
p <- add_argument(p, "--chrom-state",  help = "Roadmap core-marks chromatin-state BED (e.g. E073 dense.bed.gz)")
p <- add_argument(p, "--output-rdata", help = "Output domains.Rdata")
p <- add_argument(p, "--output-csv",   help = "Output domains.csv")
p <- add_argument(p, "--output-states", help = "Output core15States.csv")
argv <- parse_args(p)

source(argv$lib)

# Add mean/median raw counts
counts <- readH3k9acCounts(argv$counts_rds, argv$qc, argv$sample_sheet)
domains <- rowRanges(counts)
countMat <- as.matrix(assay(counts))
domains$mean   <- rowMeans2(countMat)
domains$sd     <- rowSds(countMat)
domains$median <- rowMedians(countMat)
domains$mad    <- rowMads(countMat)
rm(counts, countMat)

# Remove domains from the encode blacklist (same domains as in residuals file)
domains <- domains[!domains$blacklist]

# Read the Roadmap Epigenomic data (e.g. E073, DLPFC) and calculate the relative
# frequency of each chromatin state in each H3K9ac domain. Some domains are on
# unscaffolded contigs not in the reference epigenome and are set to NA.
epi <- import(argv$chrom_state)
seqlevelsStyle(epi) <- "Ensembl"
epi <- epi[seqnames(epi) != "MT"]  # MT has no domains
seqlevels(epi) <- seqlevelsInUse(epi)

relOv <- data.frame(domain = names(domains), stringsAsFactors = FALSE)
coreMarks <- unique(epi$name)[order(as.numeric(sapply(strsplit(unique(epi$name), "_"), "[", 1)))]
for (mark in coreMarks) {
  cm <- epi[epi$name == mark]
  is <- intersect(domains, cm)
  ov <- findOverlaps(is, domains)
  ov <- split(queryHits(ov), f = subjectHits(ov))
  basesOv <- sapply(ov, function(ind) { return(sum(width(is[ind]))) })
  relOv[, mark] <- 0
  relOv[as.integer(names(ov)), mark] <- basesOv
}
rownames(relOv) <- relOv$domain
bases <- apply(relOv[, -1], 1, sum)

# Domains without any histone-state annotation are set to NA (annotation is not
# continuous after lift-over to GRCh38, and non-standard chromosomes lack it).
ind <- bases == 0
relOv[ind, -1] <- NA
stopifnot(all(rownames(relOv) == names(domains)))
relOv <- relOv[, -1]
relOv <- relOv / bases  # divide by bases covered with histone-state annotation
mcols(domains) <- cbind(mcols(domains), relOv)

# Chromatin-state descriptions (Roadmap core 15-state model)
core15States <- read.csv(text = "StateID,State,Description,Color,ColorCode
1_TssA,TssA,Active TSS,Red,#FF0000
2_TssAFlnk,TssAFlnk,Flanking Active TSS,OrangeRed,#FF4500
3_TxFlnk,TxFlnk,Transcr. at gene 5p and 3p,LimeGreen,#32CD32
4_Tx,Tx,Strong transcription,Green,#008000
5_TxWk,TxWk,Weak transcription,DarkGreen,#006400
6_EnhG,EnhG,Genic enhancers,GreenYellow,#C2E105
7_Enh,Enh,Enhancers,Yellow,#FFFF00
8_ZNF/Rpts,ZNF/Rpts,ZNF genes & repeats,MediumAquamarine,#66CDAA
9_Het,Het,Heterochromatin,PaleTurquoise,#8A91D0
10_TssBiv,TssBiv,Bivalent/Poised TSS,IndianRed,#CD5C5C
11_BivFlnk,BivFlnk,Flanking Bivalent TSS/Enh,DarkSalmon,#E9967A
12_EnhBiv,EnhBiv,Bivalent Enhancer,DarkKhaki,#BDB76B
13_ReprPC,ReprPC,Repressed PolyComb,Silver,#808080
14_ReprPCWk,ReprPCWk,Weak Repressed PolyComb,Gainsboro,#C0C0C0
15_Quies,Quies,Quiescent/Low,White,#FFFFFF",
                          stringsAsFactors = FALSE)
write.csv(core15States, file = argv$output_states, quote = FALSE, row.names = FALSE)

# Cluster domains by chromatin-state composition (k = 7 explains > 80% of variance)
set.seed(03122014)
naInd <- apply(is.na(relOv), 1, any)
km <- kmeans(relOv[!naInd, -15], centers = 7, iter.max = 100, nstart = 20)

renameClusters <- c("1" = "C2_TssAFlnk", "2" = "C4_EnhG", "3" = "C3_Enh", "4" = "C7_Other",
                    "5" = "C5_Tx", "6" = "C6_TxWk", "7" = "C1_TssA")
relOv$Cluster <- NA
relOv$Cluster[!naInd] <- renameClusters[km$cluster]
stopifnot(all(rownames(relOv) == names(domains)))
mcols(domains)$cluster <- relOv$Cluster

save(domains, file = argv$output_rdata)
domainsDf <- as.data.frame(domains)
colnames(domainsDf) <- gsub("^X", "", colnames(domainsDf))
colnames(domainsDf)[colnames(domainsDf) == "seqnames"] <- "chr"
colnames(domainsDf)[colnames(domainsDf) == "8_ZNF.Rpts"] <- "8_ZNF/Rpts"
write.csv(domainsDf, file = argv$output_csv, row.names = FALSE, quote = FALSE)
