#' Calculate Encode QC metrics for ChIP-seq/ATAC-seq data
#'
#' This function calculates quality check parameters originally recommened
#'   by the ENCODE consortium for ChIP-seq data. Most measures are calculated
#'   including duplicated reads unless indicated otherwise by the variable's
#'   name. In case of paired-end data, the argugument "bed" should define
#'   fragments obtained from properly paired reads. For single-end data, bed
#'   file should give individual reads. Fragments/reads with low mapping
#'   quality ("multi-mapper") should be removed prior to calling this function.
#'   
#' @param sample Name of the sample.
#' @param bed Bed file defining fragments obtained from paired-end sequencing.
#' @param peaks Bed file defining peaks (significantly enriched regions).
#' @param blacklist Bed file defining the blacklisted regions excluded from the analysis.
#'
#' @return Data frame with QC metrics.
#' 
#' @author Hans
encodeMetrics <- function(sample, bed, peaks, blacklist) {

  require(rtracklayer)
  fragments <- import(bed, format="BED")
  peaks <- import(peaks, format="BED")
  blacklist <- import(blacklist, format="BED")

  # Get overall fraction of MT fragments
  mt <- sum(seqnames(fragments) == "MT") / length(fragments)
  
  # Make sure chromosome names are matching and reduce to chrs. 1-22
  stopifnot(all(seqlevelsStyle(fragments) %in% seqlevelsStyle(peaks)))
  seqlevelsStyle(blacklist) <- seqlevelsStyle(fragments)[1]
  fragments <- sort(keepSeqlevels(fragments, intersect(seqlevels(fragments), as.character(1:22)), pruning.mode="tidy"))
  peaks <- sort(keepSeqlevels(peaks, intersect(seqlevels(peaks), as.character(1:22)), pruning.mode="tidy"))
  blacklist <- sort(keepSeqlevels(blacklist, as.character(1:22), pruning.mode="tidy"))
  
  # Reduce fragments and peaks to non-blacklisted regions and initialize data frame
  f <- fragments[!overlapsAny(fragments, blacklist)]
  p <- peaks[!overlapsAny(peaks, blacklist)]
  metrics <- data.frame(SampleID=sample,
                        TotalNumFrags=length(fragments),
                        PercMt=mt,
                        PercBlacklistFrags=1-(length(f)/length(fragments)),
                        UsableFrags=length(f),
                        TotalNumPeaks=length(peaks),
                        PercBlacklistPeaks=1-(length(p)/length(peaks)),
                        UsablePeaks=length(p),
                        stringsAsFactors=FALSE)
  
  # Non-Redundant Fraction (NRF) – Number of distinct uniquely mapping reads (i.e. after removing duplicates) / Total number of reads.
  metrics$UniqueFrags <- sum(!duplicated(f))
  metrics$NRF <- metrics$UniqueFrags / metrics$UsableFrags
  
  # PBC1=M1/MDISTINCT where
  # M1: number of genomic locations where exactly one read maps uniquely
  # MDISTINCT: number of distinct genomic locations to which some read maps uniquely
  fDup <- f[duplicated(f)]
  fUnique <- f[!duplicated(f)]
  M1 <- length(fUnique) - sum(fUnique %in% fDup)
  MDISTINCT <- length(fUnique)
  metrics$PBC1 <- M1/MDISTINCT
  
  # PBC2=M1/M2 where
  # M1: number of genomic locations where only one read maps uniquely
  # M2: number of genomic locations where two reads map uniquely
  fDupDup <- fDup[duplicated(fDup)]
  M2 <- length(fDup) - sum(fDup %in% fDupDup)
  metrics$PBC2 <- M1/M2
  
  # Fraction of reads in peaks (FRiP) – Fraction of all mapped reads that fall into the called
  # peak regions, i.e. usable reads in significantly enriched peaks divided by all usable reads.
  # In general, FRiP scores correlate positively with the number of regions.
  # (Landt et al, Genome Research Sept. 2012, 22(9): 1813–1831)
  metrics$FRiP <- sum(overlapsAny(f, p)) / length(f)
  fUnique <- f[!duplicated(f)]
  metrics$UniqueFragsFRiP <- sum(overlapsAny(fUnique, p)) / length(fUnique)
  
  return(metrics)
}


# Calculates the cross correlation as implemented in epigenomix
crossCorrelation <- function(bed, blacklist, rmDup=FALSE) {
  
  require(rtracklayer)
  require(epigenomix)
  fragments <- import(bed, format="BED")
  blacklist <- import(blacklist, format="BED")

  seqlevelsStyle(blacklist) <- seqlevelsStyle(fragments)[1]
  fragments <- sort(keepSeqlevels(fragments, intersect(seqlevels(fragments), as.character(1:22)), pruning.mode="tidy"))
  blacklist <- sort(keepSeqlevels(blacklist, as.character(1:22), pruning.mode="tidy"))
  f <- fragments[!overlapsAny(fragments, blacklist)] 
  if (rmDup == TRUE) {
    f <- f[!duplicated(f)]
  }
  
  # Bed file with strand information
  if (all(c("+", "-") %in% as.character(unique(strand(f))))) {
    shift <- seq(0, 1000, 10)
    cc <- calculateCrossCorrelation(f, shift=shift, bin=10, mode="none", minReads=1, mc.cores=1)
    crossCor <- data.frame(FragmentSize=as.numeric(names(cc)),
                           CrossCor=cc)
  # Bed file without strand information
  } else {
    crossCor <- data.frame(FragmentSize=NA,
                           CrossCor=NA)
  }
  
  return(crossCor)
}


#' Annotate reads with chromatin states and calculate relative frequencies
#'
#' This function annotates each sequenced base with a chromtin state from a
#'   given reference and calculates the relative frequencies for each chromatin
#'   state. In case of paired-end data, the argugument "bed" should define
#'   fragments obtained from properly paired reads. Fragments/reads with low mapping
#'   quality ("multi-mapper") should be removed prior to calling this function.
#'   
#' @param sample Name of the sample.
#' @param bed Bed file defining fragments/reads.
#' @param chromState GRanges object with reference epigenome.
#' @param blacklist Bed file defining the blacklisted regions excluded from the analysis.
#'
#' @return Data frame with relative frequencies of chromatin states observed in the data.
#' 
#' @author Hans
chromatinStateOverlap <- function(sample, bed, chromState, blacklist, rmDup=FALSE) {

  require(rtracklayer)
  epigenome <- strsplit(basename(chromState), "_")[[1]][1]
  fragments <- import(bed, format="BED")
  chromState <- import(chromState, format="BED")
  blacklist <- import(blacklist, format="BED")
  
  # Make sure chromosome names are matching and reduce to chrs. 1-22
  seqlevelsStyle(chromState) <- seqlevelsStyle(fragments)[1]
  seqlevelsStyle(blacklist) <- seqlevelsStyle(fragments)[1]
  fragments <- sort(keepSeqlevels(fragments, intersect(seqlevels(fragments), as.character(1:22)), pruning.mode="tidy"))
  chromState <- sort(keepSeqlevels(chromState, as.character(1:22), pruning.mode="tidy"))
  blacklist <- sort(keepSeqlevels(blacklist, as.character(1:22), pruning.mode="tidy"))
  
  # Remove duplicates
  if (rmDup == TRUE) {
    fragments <- fragments[!duplicated(fragments)]
  }

  # Remove blacklisted regions from epigenome
  c <- chromState[!overlapsAny(chromState, blacklist)]
  stopifnot(isDisjoint(c))
  
  # Proportion of chromatin states in reference epigenome
  reference <- sapply(split(c, mcols(c)$name), function (cs) {
    return(sum(as.numeric(width(cs))))
  })
  reference <- reference / sum(as.numeric(width(c)))
  
  # Proportion of chromatin states sequenced in sample
  ov <- findOverlaps(fragments, c)
  totalBases <- sum(as.numeric(width(pintersect(fragments[queryHits(ov)], c[subjectHits(ov)]))))
  chromStateBases <- sapply(split(c, mcols(c)$name), function (cs) {
    ov <- findOverlaps(fragments, cs)
    return(sum(as.numeric(width(pintersect(fragments[queryHits(ov)], cs[subjectHits(ov)])))))
  })
  chromStateBases <- chromStateBases / totalBases
  
  order <- c("1_TssA", "2_TssAFlnk", "3_TxFlnk", "4_Tx", "5_TxWk",
             "6_EnhG", "7_Enh", "8_ZNF/Rpts", "9_Het", "10_TssBiv",
             "11_BivFlnk", "12_EnhBiv", "13_ReprPC", "14_ReprPCWk", "15_Quies")
  results <- data.frame(rbind(chromStateBases[order], reference[order]), stringsAsFactors=FALSE)
  colnames(results) <- order
  results$SampleID <- c(sample, "Reference")
  results$Epigenome <- epigenome
  results <- results[, c(16, 17, 1:15)]
  
  return(results)
}
