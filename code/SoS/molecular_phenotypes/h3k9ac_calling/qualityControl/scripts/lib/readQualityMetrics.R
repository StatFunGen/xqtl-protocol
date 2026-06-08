readPicard.alignment_summary_metrics <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*.alignment_summary_metrics\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".alignment_summary_metrics", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub(".alignment_summary_metrics", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", fill=TRUE, stringsAsFactors=FALSE)
    if (all(c("FIRST_OF_PAIR", "SECOND_OF_PAIR") %in% m$CATEGORY) & !"UNPAIRED" %in% m$CATEGORY) {
      ind <- m$CATEGORY %in% c("FIRST_OF_PAIR", "SECOND_OF_PAIR")
    } else if ("UNPAIRED" %in% m$CATEGORY & !any(c("FIRST_OF_PAIR", "SECOND_OF_PAIR") %in% m$CATEGORY)) {
      ind <- m$CATEGORY == "UNPAIRED"
    } else {
      stop("Could not detect whether library is paired or unpaired.")
    }
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               PF_READS=sum(as.numeric(m$PF_READS[ind])),
                               PF_READS_ALIGNED=sum(as.numeric(m$PF_READS_ALIGNED[ind])),
                               PCT_PF_READS_ALIGNED=sum(as.numeric(m$PF_READS_ALIGNED[ind]))/sum(as.numeric(m$PF_READS[ind])),
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readPicard.rna_metrics <- function(source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*.rna_metrics\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".rna_metrics", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub(".rna_metrics", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", stringsAsFactors=FALSE, nrows=1)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               PCT_RIBOSOMAL_BASES=m$PCT_RIBOSOMAL_BASES,
                               PCT_CODING_BASES=m$PCT_CODING_BASES,
                               PCT_UTR_BASES=m$PCT_UTR_BASES,
                               PCT_INTRONIC_BASES=m$PCT_INTRONIC_BASES,
                               PCT_INTERGENIC_BASES=m$PCT_INTERGENIC_BASES,
                               PCT_MRNA_BASES=m$PCT_MRNA_BASES,
                               PCT_USABLE_BASES=m$PCT_USABLE_BASES,
                               MEDIAN_CV_COVERAGE=m$MEDIAN_CV_COVERAGE,
                               MEDIAN_5PRIME_BIAS=m$MEDIAN_5PRIME_BIAS,
                               MEDIAN_3PRIME_BIAS=m$MEDIAN_3PRIME_BIAS,
                               MEDIAN_5PRIME_TO_3PRIME_BIAS=m$MEDIAN_5PRIME_TO_3PRIME_BIAS,
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readPicard.duplicate_metrics <- function(source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*.duplicate_metrics\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".duplicate_metrics", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub(".duplicate_metrics", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", stringsAsFactors=FALSE, nrows=1)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               PERCENT_DUPLICATION=m$PERCENT_DUPLICATION,
                               ESTIMATED_LIBRARY_SIZE=m$ESTIMATED_LIBRARY_SIZE,
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readPicard.wgs_metrics <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*.wgs_metrics\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".wgs_metrics", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub(".wgs_metrics", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", stringsAsFactors=FALSE, nrows=1)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               MEDIAN_COVERAGE=m$MEDIAN_COVERAGE,
                               MAD_COVERAGE=m$MAD_COVERAGE,
                               PCT_EXC_DUPE=m$PCT_EXC_DUPE,
                               PCT_EXC_TOTAL=m$PCT_EXC_TOTAL,
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readPicard.insert_size_metrics <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*.insert_size_metrics\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".insert_size_metrics", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub(".insert_size_metrics", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", stringsAsFactors=FALSE, nrows=1)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               MEDIAN_INSERT_SIZE=m$MEDIAN_INSERT_SIZE,
                               MODE_INSERT_SIZE=m$MODE_INSERT_SIZE,
                               MEDIAN_ABSOLUTE_DEVIATION=m$MEDIAN_ABSOLUTE_DEVIATION,
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readPicard.gc_bias.summary_metrics <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*.gc_bias.summary_metrics\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".gc_bias.summary_metrics", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub(".gc_bias.summary_metrics", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", stringsAsFactors=FALSE, nrows=1)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               AT_DROPOUT=m$AT_DROPOUT,
                               GC_DROPOUT=m$GC_DROPOUT,
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readPicard <- function(source) {
  metrics.aln <- readPicard.alignment_summary_metrics(source)
  metrics.rna <- readPicard.rna_metrics(source)
  metrics.dup <- readPicard.duplicate_metrics(source)

  stopifnot(all(row.names(metrics.aln) %in% row.names(metrics.rna)) &
            all(row.names(metrics.rna) %in% row.names(metrics.dup)) &
            all(row.names(metrics.dup) %in% row.names(metrics.aln)))
  
  metrics.aln$File <- NULL
  metrics.rna$File <- NULL
  metrics.dup$File <- NULL
  metrics.rna$Sample <- NULL
  metrics.dup$Sample <- NULL
  
  metrics <- cbind(metrics.aln, metrics.rna[row.names(metrics.aln), ])
  metrics <- cbind(metrics, metrics.dup[row.names(metrics.aln), ])
  
  return(metrics)
}


readRSEM.cnt <- function (source) {
  
  # RSEM .cnt files gives statistics about the (transcriptome) alignment passed to RSEM:
  # Row 1: N0 (# unalignable reads);
  #        N1 (# alignable reads);
  #        N2 (# filtered reads due to too many alignments);
  #        N_tot (N0+N1+N2)
  # Row 2: nUnique (# reads aligned uniquely to a gene);
  #        nMulti (# reads aligned to multiple genes);
  #        nUncertain (# reads aligned to multiple locations in the given reference sequences, which include isoform-level multi-mapping reads)
  # Row 3: nHits (# total alignments);
  #        read_type (0: single-end read, no quality; 1: single-end read, with quality score; 2: paired-end read, no quality score; 3: paired-end read, with quality score)
  # Source: https://groups.google.com/forum/#!topic/rsem-users/usmPKgsC5LU
  # Note: N1 = nUnique + nMulti
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*_rsem.cnt\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub("_rsem.cnt", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub("_rsem.cnt", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=FALSE, sep=" ", comment.char="#", stringsAsFactors=FALSE, nrows=3, fill=TRUE)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               TotalReads=m[1, 4],
                               AlignedReads=m[1, 2],
                               UniquelyAlignedReads=m[2, 1],
                               stringsAsFactors=FALSE)
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readMACS2 <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- system(paste("find -L", source, "-name \"*_peaks.xls\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub("_peaks.xls", "", basename(files), fixed=TRUE)
  } else {
    files <- source
    samples <- gsub("_peaks.xls", "", basename(files), fixed=TRUE)
  }
  
  metrics <- list()
  for (i in 1:length(files)) {
    m <- read.table(files[i], header=TRUE, sep="\t", comment.char="#", stringsAsFactors=FALSE)
    metrics[[i]] <- data.frame(Sample=samples[i],
                               File=files[i],
                               MacsFragSize=NA,
                               NumberPeaks=nrow(m),
                               MeanWidth=mean(m$length),
                               MedianWidth=median(m$length),
                               stringsAsFactors=FALSE)
    header <- readLines(files[i], n=100)
    ind <- grep("^# d = ", header, fixed=FALSE)
    stopifnot(length(ind) == 1)
    metrics[[i]]$MacsFragSize <- as.numeric(gsub("^# d = ", "", header[ind]))
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  
  return(metrics)
}


readEncodeMetrics <- function (source, paired=FALSE) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  stopifnot(file.info(source)$isdir)
  
  # Read in metrics first
  files <- system(paste("find -L", source, "-name \"*_metrics.csv\""), intern=TRUE)
  stopifnot(length(files) > 0)
  samples <- gsub("_metrics.csv", "", basename(files), fixed=TRUE)
  
  metrics <- list()
  for (i in 1:length(files)) {
    metrics[[i]] <- read.table(files[i], header=TRUE, sep=",", stringsAsFactors=FALSE, colClasses=c(SampleID="character"))
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$SampleID
  

  # Read cross-correlation (if single end data)
  if (paired == FALSE) {
    files <- system(paste("find -L", source, "-name \"*_crossCor.csv\""), intern=TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub("_crossCor.csv", "", basename(files), fixed=TRUE)
  
    cc <- list()
    for (i in 1:length(files)) {
      crosscor <- read.table(files[i], header=TRUE, sep=",", stringsAsFactors=FALSE)
      ind <- which.max(crosscor$CrossCor)
      if (length(ind) == 1) {
        cc[[i]] <- data.frame(Sample=samples[i],
                              CrossCor=crosscor$CrossCor[ind],
                              CCFragSize=crosscor$FragmentSize[ind])
      } else {
        cc[[i]] <- data.frame(Sample=samples[i],
                              CrossCor=NA,
                              CCFragmentSize=NA)
      }
    }
    cc <- do.call(rbind, cc)
    row.names(cc) <- cc$Sample
    
    stopifnot(all(rownames(metrics) %in% rownames(cc)) & all(rownames(cc) %in% rownames(metrics)))
    cc <- cc[rownames(metrics), ]
    cc$Sample <- NULL
    metrics <- cbind(metrics, cc)
  }
  
  return(metrics)
}


readRNAseqQC <- function(sourcePicard, sourceRSEM) {
  metrics.picard <- readPicard(sourcePicard)
  metrics.rsem <- readRSEM.cnt(sourceRSEM)
  
  stopifnot(all(row.names(metrics.picard) %in% row.names(metrics.rsem)) &
              all(row.names(metrics.rsem) %in% row.names(metrics.picard)))
  
  metrics.rsem$File <- NULL
  metrics.rsem$Sample <- NULL
  
  metrics <- cbind(metrics.picard, metrics.rsem[row.names(metrics.picard), ])
  
  return(metrics)
}


readChIPseqQC <- function(sourcePicard, sourceMacs2, paired=FALSE) {
  
  # Alignment metrics
  metrics.aln <- readPicard.alignment_summary_metrics(sourcePicard)
  metrics.gc <- readPicard.gc_bias.summary_metrics(sourcePicard)
  metrics.dup <- readPicard.duplicate_metrics(sourcePicard)
  if (paired) {
    metrics.ins <- readPicard.insert_size_metrics(sourcePicard)
  }
  
  stopifnot(all(row.names(metrics.aln) %in% row.names(metrics.gc)) &
            all(row.names(metrics.gc) %in% row.names(metrics.dup)) &
            all(row.names(metrics.dup) %in% row.names(metrics.aln)))
  if (paired) {
    stopifnot(all(row.names(metrics.aln) %in% row.names(metrics.ins)) &
              all(row.names(metrics.ins) %in% row.names(metrics.aln)))
  }
  
  metrics.aln$File <- NULL
  metrics.gc$File <- NULL
  metrics.dup$File <- NULL
  metrics.gc$Sample <- NULL
  metrics.dup$Sample <- NULL
  if (paired) {
    metrics.ins$File <- NULL
    metrics.ins$Sample <- NULL
  }
  
  metrics <- cbind(metrics.aln, metrics.gc[row.names(metrics.aln), ])
  metrics <- cbind(metrics, metrics.dup[row.names(metrics.aln), ])
  if (paired) {
    metrics <- cbind(metrics, metrics.ins[row.names(metrics.aln), ])
  }
  
  # ChIP-seq metrics
  macs <- readMACS2(sourceMacs2)
  encode <- readEncodeMetrics(sourceMacs2, paired=paired)
  
  stopifnot(all(row.names(macs) %in% row.names(encode)) & all(row.names(encode) %in% row.names(macs)))
  macs$Sample <- NULL
  macs$File <- NULL
  macs <- macs[rownames(encode), ]
  encode <- cbind(encode, macs)
  
  stopifnot(all(row.names(encode) %in% row.names(metrics)))
  metrics <- merge(metrics, encode, by="row.names", all.x=TRUE)
  metrics$Row.names <- NULL
  metrics$SampleID <- NULL
  
  return(metrics)
}

#' Aggregate chromatin state relative frequencies
#'
#' @description 
#' This function reads in the proportion of chromatin states sequenced generated in step 7.
#' Relative frequencies are returned as data frame. If multiple epigenomes were used,
#' all results are combined into one data frame.
#' 
#' @param peaksDir MACS2 directory with chromatin state frequency files.
#'
#' @return Data frame with relative frequencies or NA if no files are found.
#'
#' @author Hans
aggregateChromatinStateFrequencies <- function(peaksDir) {
  
  files <- system(paste("find", peaksDir, "-name \"*_chromStates.csv\""), intern=TRUE)
  if (length(files) > 0) {
    chromatinMetrics <- list()
    for (i in 1:length(files)) {
      chromatinMetrics[[i]] <- read.csv(files[i], check.names=FALSE, stringsAsFactors=FALSE)
    }
    chromatinMetrics <- do.call(rbind, chromatinMetrics)
    chromatinMetrics <- chromatinMetrics[!(duplicated(chromatinMetrics)), ]
    chromatinMetrics <- chromatinMetrics[order(chromatinMetrics$SampleID, chromatinMetrics$Epigenome), ]
    
    return(chromatinMetrics)
    
  } else {
    return(NA)
  }
}
