suppressPackageStartupMessages({
  library(argparser)
})

p <- arg_parser("Aggregate ChIP-seq QC metrics (Picard + MACS2 + ENCODE) into per-sample tables")
p <- add_argument(p, "--lib",              help = "Path to readQualityMetrics.R")
p <- add_argument(p, "--picard-dir",       help = "Directory with Picard metric files (bowtie2Aligned)")
p <- add_argument(p, "--macs2-dir",        help = "Directory with MACS2 + ENCODE metric files (macs2Peaks)")
p <- add_argument(p, "--output-qc",        help = "Output qualityMetrics.csv")
p <- add_argument(p, "--output-chromatin", help = "Output chromatinMetrics.csv")
p <- add_argument(p, "--paired",           help = "Paired-end data", flag = TRUE)
argv <- parse_args(p)

source(argv$lib)

# NOTE: qualityMetrics.csv is written with write.table (space-delimited, headered,
# no row names) because that is the exact format readH3k9acCounts.R expects.
qualityMetrics <- readChIPseqQC(sourcePicard = argv$picard_dir,
                                sourceMacs2  = argv$macs2_dir,
                                paired       = argv$paired)
write.table(qualityMetrics, file = argv$output_qc,
            col.names = TRUE, row.names = FALSE, quote = FALSE)

chromatinMetrics <- aggregateChromatinStateFrequencies(peaksDir = argv$macs2_dir)
if (!is.na(chromatinMetrics)[1]) {
  write.table(chromatinMetrics, file = argv$output_chromatin,
              col.names = TRUE, row.names = FALSE, quote = FALSE)
}
