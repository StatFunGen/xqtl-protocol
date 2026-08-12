readPicard.alignment_summary_metrics <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- list.files(source, pattern = "\\.alignment_summary_metrics$", recursive = TRUE, full.names = TRUE)
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


readPicard.duplicate_metrics <- function(source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- list.files(source, pattern = "\\.duplicate_metrics$", recursive = TRUE, full.names = TRUE)
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


readPicard.insert_size_metrics <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- list.files(source, pattern = "\\.insert_size_metrics$", recursive = TRUE, full.names = TRUE)
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
    files <- list.files(source, pattern = "\\.gc_bias\\.summary_metrics$", recursive = TRUE, full.names = TRUE)
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


readMACS2 <- function (source) {
  
  stopifnot(length(source) == 1)
  stopifnot(file.exists(source))
  isDir <- file.info(source)$isdir
  if (isDir) {
    files <- list.files(source, pattern = "_peaks\\.xls$", recursive = TRUE, full.names = TRUE)
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
  files <- list.files(source, pattern = "_metrics\\.csv$", recursive = TRUE, full.names = TRUE)
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
    files <- list.files(source, pattern = "_crossCor\\.csv$", recursive = TRUE, full.names = TRUE)
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
  
  files <- list.files(peaksDir, pattern = "_chromStates\\.csv$", recursive = TRUE, full.names = TRUE)
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
