library(GenomicRanges)
library(rtracklayer)

# R command line argument
args <- commandArgs(trailingOnly=TRUE)
sample <- args[1]
print(sample)

# Read domains
domains <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/H3K9acDomains.csv",
                    header=TRUE, stringsAsFactors=FALSE)
domains <- GRanges(IRanges(start=domains$start, end=domains$end),
                   seqnames=domains$chr,
                   name=domains$name)
names(domains) <- domains$name

# Read fragment size
qc <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/qualityMetrics.csv",
                 sep=" ", header=TRUE, stringsAsFactors=FALSE)
fragSize <- qc$CCFragSize[qc$Sample == sample]

# Read fragments (low mapping quality already filtered, but duplicates still included)
fragments <- import(paste("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/", sample, "/", sample, ".bed", sep=""))
fragments <- fragments[!duplicated(fragments)]
fragments <- resize(fragments, fragSize, fix="start")

# Count overlaps and save
counts <- countOverlaps(domains, fragments)
save(counts, file=paste("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/tmp/", sample, ".RData", sep=""))