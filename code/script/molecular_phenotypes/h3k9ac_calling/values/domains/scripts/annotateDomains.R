library(rtracklayer)
source("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/scripts/readH3k9acCounts.R")

# Add mean/median raw counts
counts <- readH3k9acCounts()
domains <- rowRanges(counts)
domains$mean <- apply(assay(counts), 1, mean)
domains$sd <- apply(assay(counts), 1, sd)
domains$median <- apply(assay(counts), 1, median)
domains$mad <- apply(assay(counts), 1, mad)
rm(counts)


# Remove domains from the encode blacklist (same domains as in residuals file)
domains <- domains[!domains$blacklist] # 92076


# Read in the Roadmap Epigenomic data from E073 (DLPFC) and calculate
# the relative frequency of each chromatin state in each H3K9ac domain.
# Some domains are located on unscaffolded contigs not included
# in the reference epigenome and are set to NA.
epi <- import("/mnt/mfs/ctcn/resources/chromatinStates/coreMarks_15/hg38/E073_15_coreMarks_hg38lift_dense.bed.gz")
seqlevelsStyle(epi) <- "Ensembl"
epi <- epi[seqnames(epi) != "MT"] # MT has no domains
seqlevels(epi) <- seqlevelsInUse(epi)

relOv <- data.frame(domain=names(domains), stringsAsFactors=FALSE)
coreMarks <- unique(epi$name)[order(as.numeric(sapply(strsplit(unique(epi$name), "_"), "[", 1)))]
for (mark in coreMarks) {
  cm <- epi[epi$name == mark]
  is <- intersect(domains, cm)
  ov <- findOverlaps(is, domains)
  ov <- split(queryHits(ov), f=subjectHits(ov))
  basesOv <- sapply(ov, function(ind) {return(sum(width(is[ind])))})
  relOv[, mark] <- 0
  relOv[as.integer(names(ov)), mark] <- basesOv
  print(paste(mark, "done."))
}
rownames(relOv) <- relOv$domain
bases <- apply(relOv[, -1], 1, sum)
sum(bases == width(domains)) # 91829
# Most but not all peaks are completely covered with annotated histone states
# because annotation is not continuous anymore after lift over to GRCh38.
# Further, non standard chromosomes are missing annotation (in hg19 version, too)

ind <- bases != width(domains)
table(as.character(seqnames(domains[ind])))
#  1         10         12         13         15         16         17         19          2         20         22          3          4          6          7 
# 75         26          1          3          1          1         41          5          2         13          2          5          2          3         21 
#  8          9 GL000194.1 GL000195.1 GL000205.2 GL000219.1 KI270713.1 KI270733.1          X 
# 15         12          1          2          2          3          2          1          8
ind <- bases == 0
table(as.character(seqnames(domains[ind])))
#          1         10         13         15         16         17         20         22          3          4          6          7          8          9 GL000194.1 
#         60         18          3          1          1         31         12          1          1          1          3         18          7          7          1 
# GL000195.1 GL000205.2 GL000219.1 KI270713.1 KI270733.1          X 
#          2          2          3          2          1          1 

# GL000192.1 GL000194.1 GL000195.1 GL000204.1 GL000205.1 GL000219.1 GL000220.1 GL000224.1 
#          3          1          4         12          5          3          9          1 

# We set domains without any annotation to NA
relOv[ind, -1] <- NA
stopifnot(all(rownames(relOv) == names(domains)))
relOv <- relOv[, -1]
relOv <- relOv / bases # divide by bases covered with histone state annotation
mcols(domains) <- cbind(mcols(domains), relOv)


# Generate csv file describing the chromatin states
core15States <- read.csv(text="StateID,State,Description,Color,ColorCode
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
write.csv(core15States, file="core15States.csv", quote=FALSE, row.names=FALSE)


# Cluster domains using k means
# Find optimal k
ss <- data.frame(k=1:10, SSwithin=c(1, rep(NA, 9)))
naInd <- apply(is.na(relOv), 1, any)
for (i in 2:10) {
  set.seed(03122014)
  km <- kmeans(relOv[!naInd, -15], centers=i, iter.max=100, nstart=20)
  ss$SSwithin[i] <- km$tot.withinss / km$totss
}
plot(x=ss$k, ss$SSwithin)
ss # k=7 explains > 80% (and gives meaningful cluster annotation)
#     k  SSwithin
# 1   1 1.0000000
# 2   2 0.7086738
# 3   3 0.4550726
# 4   4 0.3290877
# 5   5 0.2688309
# 6   6 0.2187694
# 7   7 0.1742803
# 8   8 0.1510186
# 9   9 0.1329511
# 10 10 0.1206515

set.seed(03122014)
naInd <- apply(is.na(relOv), 1, any)
km <- kmeans(relOv[!naInd, -15], centers=7, iter.max=100, nstart=20)

calcCenters <- function(relOv, cluster) {
  centerList <- lapply(split.data.frame(relOv, f=cluster), apply, 2, mean)
  return(do.call(rbind, centerList))
}

plotCenters <- function(centers) {
  require(ggplot2)
  require(reshape2)
  centers <- melt(centers)
  ggplot(centers, aes(x=Var1, y=value, fill=Var2)) +
    geom_bar(stat="identity") + xlab("") +
    ylab("Mean relative overlap with core mark") +
    scale_fill_discrete(name="Core mark") +
    theme(axis.text.x = element_text(angle = 90))
}

plotCenters(calcCenters(relOv[!naInd, ], km$cluster))
km$centers
#         1_TssA   2_TssAFlnk     3_TxFlnk         4_Tx      5_TxWk      6_EnhG       7_Enh   8_ZNF/Rpts        9_Het
# 1 2.751118e-01 0.4529172082 6.747244e-03 0.0025092883 0.052300940 0.010978231 0.161091607 3.512045e-05 1.972245e-05
# 2 1.691552e-03 0.0066369010 6.316768e-03 0.0675349435 0.056112436 0.824455004 0.037167876 0.000000e+00 0.000000e+00
# 3 4.581850e-03 0.0164557243 7.413255e-05 0.0005512967 0.081651568 0.005129792 0.858599741 3.010692e-06 1.356437e-05
# 4 1.517269e-02 0.0037367869 4.952942e-03 0.0016219070 0.009090886 0.002942837 0.028455103 1.135177e-03 1.132996e-03
# 5 8.902234e-05 0.0001480798 3.018007e-04 0.9348239088 0.035217697 0.028379842 0.001039649 0.000000e+00 0.000000e+00
# 6 5.166399e-03 0.0026572525 1.428978e-04 0.0058999576 0.919242974 0.004211011 0.060033951 1.135431e-04 2.424043e-05
# 7 7.798077e-01 0.0805575944 3.331885e-03 0.0003354711 0.034110074 0.001677667 0.035136399 6.003638e-05 6.492951e-06
# 10_TssBiv   11_BivFlnk    12_EnhBiv    13_ReprPC  14_ReprPCWk
# 1 1.704583e-03 3.879849e-03 1.354797e-03 2.636850e-04 0.0071377570
# 2 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.0000000000
# 3 0.000000e+00 7.227936e-05 5.959586e-04 2.232781e-05 0.0037362948
# 4 8.003660e-02 2.828201e-02 2.262568e-02 2.088520e-02 0.1111400371
# 5 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.0000000000
# 6 3.113551e-05 3.608791e-05 2.883861e-05 3.183603e-06 0.0002910542
# 7 7.801406e-03 3.585285e-03 6.850914e-04 4.480636e-04 0.0064245948

table(km$cluster)
#    1     2     3     4     5     6     7 
# 9649  3587 24802 16375  4478 17719 15290 

renameClusters <- c("1"="C2_TssAFlnk", "2"="C4_EnhG", "3"="C3_Enh", "4"="C7_Other", "5"="C5_Tx", "6"="C6_TxWk", "7"="C1_TssA")
relOv$Cluster <- NA
stopifnot(all(rownames(relOv[!ind, ]) == names(km$cluster)))
relOv$Cluster[!ind] <- renameClusters[km$cluster]
stopifnot(all(rownames(relOv) == names(domains)))
mcols(domains)$cluster <- relOv$Cluster
table(domains$cluster, useNA="always")
# C1_TssA C2_TssAFlnk      C3_Enh     C4_EnhG       C5_Tx     C6_TxWk    C7_Other        <NA>
#   15290        9649       24802        3587        4478       17719       16375         176

save(domains, file="domains.Rdata")
domainsDf <- as.data.frame(domains)
colnames(domainsDf) <- gsub("^X", "", colnames(domainsDf))
colnames(domainsDf)[colnames(domainsDf) == "seqnames"] <- "chr"
colnames(domainsDf)[colnames(domainsDf) == "8_ZNF.Rpts"] <- "8_ZNF/Rpts"
write.csv(domainsDf, file="domains.csv",
          row.names=FALSE, quote=FALSE)