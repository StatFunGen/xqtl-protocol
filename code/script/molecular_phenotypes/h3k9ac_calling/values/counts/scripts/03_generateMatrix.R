suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
  library(GenomicRanges)
  library(SummarizedExperiment)
})

p <- arg_parser("Assemble per-sample H3K9ac fragment counts into a SummarizedExperiment")
p <- add_argument(p, "--sample-sheet", help = "sampleSheet.csv")
p <- add_argument(p, "--domains",      help = "H3K9acDomains.csv")
p <- add_argument(p, "--counts-dir",   help = "Directory with per-sample <sample>.RData count vectors")
p <- add_argument(p, "--output",       help = "Output H3K9acCounts.rds")
argv <- parse_args(p)

sampleSheet <- argv$sample_sheet
domainsCsv  <- argv$domains
countsDir   <- argv$counts_dir
outRds      <- argv$output

samples <- vroom(sampleSheet, show_col_types = FALSE) %>% filter(Quality == "Pass")

domainsDf <- vroom(domainsCsv, show_col_types = FALSE)
domains <- GRanges(
  seqnames = domainsDf$chr,
  ranges   = IRanges(start = domainsDf$start, end = domainsDf$end),
  strand   = domainsDf$strand,
  log10p         = domainsDf$log10p,
  log10q         = domainsDf$log10q,
  foldEnrichment = domainsDf$foldEnrichment,
  pileup         = domainsDf$pileup,
  blacklist      = domainsDf$blacklist,
  name           = domainsDf$name)
names(domains) <- domains$name

# One column per Pass sample, from the per-sample count vectors written by 02_countFragments.R
countMat <- vapply(samples$SampleID, function(s) {
  e <- new.env()
  load(file.path(countsDir, paste0(s, ".RData")), envir = e)
  cnts <- get("counts", envir = e)
  stopifnot(all(names(cnts) == domainsDf$name))
  cnts
}, numeric(nrow(domainsDf)))
colnames(countMat) <- samples$SampleID
rownames(countMat) <- domainsDf$name

se <- SummarizedExperiment(
  assays    = list(counts = countMat),
  rowRanges = domains,
  colData   = DataFrame(samples, row.names = samples$SampleID))
saveRDS(se, outRds)
