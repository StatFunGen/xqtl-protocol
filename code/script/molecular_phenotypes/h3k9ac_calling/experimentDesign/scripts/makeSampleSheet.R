suppressPackageStartupMessages({
  library(argparser)
})

p <- arg_parser("Build the experiment sample sheet (SampleID, Batch, Quality). Dataset-specific setup.")
p <- add_argument(p, "--exp-summary", help = "ChIP-seq data summary TSV")
p <- add_argument(p, "--qc-file",     help = "Nature Neuroscience QualityControl TSV (SampleID, Batch)")
p <- add_argument(p, "--source-dir",  help = "Directory of source .bam files (existence check)")
p <- add_argument(p, "--output",      help = "Output sampleSheet.csv")
argv <- parse_args(p)

# NOTE: kept on base read.csv (not vroom) because the column references below rely
# on read.csv's name mangling (e.g. "ChIP Batch" -> "ChIP.Batch"); this is one-off,
# dataset-specific setup, not a reusable per-sample worker.
expSheet <- read.csv(argv$exp_summary, sep = "\t", stringsAsFactors = FALSE)

# Remove failed / duplicate samples
expSheet <- expSheet[expSheet$ProjID != "66754397", ]
expSheet <- expSheet[!(expSheet$ProjID == "11464261" & expSheet$Pool == "8-Plex"), ]

natNeuro <- read.csv(argv$qc_file, sep = "\t", stringsAsFactors = FALSE,
                     colClasses = c("SampleID" = "character"))
stopifnot(all(natNeuro$SampleID %in% expSheet$ProjID))

# Add QC filter from the Nature Neuroscience paper
ind <- match(natNeuro$SampleID, expSheet$ProjID)
stopifnot(all(expSheet$ChIP.Batch[ind] == natNeuro$Batch))
expSheet$Quality <- "Fail"
expSheet$Quality[ind] <- "Pass"

# Match names of controls to bam files
expSheet$ProjID[expSheet$ProjID == "Positive Control"] <- c("PC-Pool-1_8plex", "PC-Pool-2_8plex")
expSheet$ProjID[expSheet$ProjID == "Negative Control"] <- c("NC-Pool-1_8plex", "NC-Pool-2_8plex")

sampleSheet <- expSheet[, c(1, 5, 19)]
colnames(sampleSheet) <- c("SampleID", "Batch", "Quality")

# Check that all bam files exist
stopifnot(all(file.exists(file.path(argv$source_dir, paste0(sampleSheet$SampleID, ".bam")))))

write.table(sampleSheet, file = argv$output, row.names = FALSE, sep = ",", quote = FALSE)
