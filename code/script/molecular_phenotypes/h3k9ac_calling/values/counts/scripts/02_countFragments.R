suppressPackageStartupMessages({
  library(argparser)
  library(GenomicRanges)
  library(rtracklayer)
})

p <- arg_parser("Count H3K9ac fragment overlaps per domain for one sample")
p <- add_argument(p, "--sample",  help = "Sample ID (matched against the QC table's Sample column)")
p <- add_argument(p, "--domains", help = "H3K9ac domains CSV (columns: chr, start, end, name)")
p <- add_argument(p, "--qc",      help = "Quality-metrics CSV (space-separated; Sample, CCFragSize columns)")
p <- add_argument(p, "--bed",     help = "Fragment BED file for this sample")
p <- add_argument(p, "--output",  help = "Output .RData path for the per-domain counts vector")
argv <- parse_args(p)

# Domains -> GRanges
domains <- read.csv(argv$domains, header = TRUE, stringsAsFactors = FALSE)
domains <- GRanges(IRanges(start = domains$start, end = domains$end),
                   seqnames = domains$chr, name = domains$name)
names(domains) <- domains$name

# Fragment size for this sample
qc <- read.table(argv$qc, sep = " ", header = TRUE, stringsAsFactors = FALSE)
fragSize <- qc$CCFragSize[qc$Sample == argv$sample]

# Fragments (low-MAPQ already filtered upstream; duplicates still present)
fragments <- import(argv$bed)
fragments <- fragments[!duplicated(fragments)]
fragments <- resize(fragments, fragSize, fix = "start")

# Count overlaps and save
counts <- countOverlaps(domains, fragments)
dir.create(dirname(argv$output), recursive = TRUE, showWarnings = FALSE)
save(counts, file = argv$output)
