#!/usr/bin/env Rscript
# Builds the Torino fixture: single-end RNA-seq alignments for 24 samples over
# AP000547.3 (chr22), simulated from three of its Ensembl 103 transcripts mixed in
# proportions that vary across samples. See README.md in this directory.
#
# Needs base R, optparse and samtools on PATH. Deterministic for a given --seed.
suppressPackageStartupMessages(library(optparse))

opt <- parse_args(OptionParser(option_list = list(
    make_option("--fasta", help = "chr22 FASTA (bgzipped or plain) covering AP000547.3"),
    make_option("--outdir", help = "fixture directory to write into"),
    make_option("--n-samples", dest = "n_samples", type = "integer", default = 24L),
    make_option("--read-length", dest = "read_length", type = "integer", default = 100L),
    make_option("--mean-reads", dest = "mean_reads", type = "integer", default = 1500L),
    make_option("--seed", type = "integer", default = 20261110L)
)))
for (a in c("fasta", "outdir")) if (is.null(opt[[a]])) stop("--", a, " is required")
if (Sys.which("samtools") == "") stop("samtools is not on PATH")
set.seed(opt$seed)

chrom <- "chr22"
# Exons (1-based, inclusive) of three AP000547.3 transcripts, from the reference GTF in
# ../leafcutter/reference. ENST00000592918 keeps the last intron of ENST00000592107
# (16614179-16614698) inside its final exon; ENST00000591299 ends in an alternative
# last exon that starts upstream of both.
isoforms <- list(
    spliced          = rbind(c(16601948, 16602215), c(16611658, 16611893),
                             c(16614077, 16614178), c(16614699, 16615111)),
    intron_retention = rbind(c(16601911, 16602215), c(16611658, 16611893),
                             c(16614077, 16615111)),
    alt_last_exon    = rbind(c(16602050, 16602215), c(16611658, 16611893),
                             c(16613985, 16614333))
)

# ---- reference sequence for the gene window
read_fasta_chrom <- function(path, name) {
    con <- if (grepl("\\.gz$", path)) gzfile(path) else file(path)
    on.exit(close(con))
    lines <- readLines(con)
    hdr <- grep("^>", lines)
    i <- which(sub("^>(\\S+).*", "\\1", lines[hdr]) == name)
    if (length(i) != 1) stop(name, " not found in ", path)
    last <- if (i < length(hdr)) hdr[i + 1] - 1 else length(lines)
    paste(lines[(hdr[i] + 1):last], collapse = "")
}
genome <- read_fasta_chrom(opt$fasta, chrom)
window <- range(unlist(isoforms))
if (grepl("N", substr(genome, window[1], window[2]))) stop("FASTA is masked over the gene window")

# ---- per-sample isoform proportions: two groups that differ in intron retention,
# plus a continuous spread in alternative last exon usage across all samples
n <- opt$n_samples
group <- rep(c("A", "B"), length.out = n)
ir <- ifelse(group == "A", 0.15, 0.55) + runif(n, -0.08, 0.08)
ale <- seq(0.05, 0.40, length.out = n)[sample(n)]
props <- cbind(spliced = 1 - ir - ale, intron_retention = ir, alt_last_exon = ale)
props[props < 0.02] <- 0.02
props <- props / rowSums(props)
lib <- pmax(200L, rpois(n, opt$mean_reads * runif(n, 0.6, 1.4)))

# ---- one read from a transcript: start position on the transcript -> genomic POS + CIGAR
read_from <- function(ex, len) {
    widths <- ex[, 2] - ex[, 1] + 1
    tlen <- sum(widths)
    s <- sample.int(tlen - len + 1, 1)          # 1-based start on the transcript
    cum <- c(0, cumsum(widths))
    k <- max(which(cum < s))                     # exon holding the start
    pos <- ex[k, 1] + (s - cum[k] - 1)
    ops <- character(0); left <- len; at <- pos; seq <- character(0)
    while (left > 0) {
        take <- min(left, ex[k, 2] - at + 1)
        ops <- c(ops, paste0(take, "M"))
        seq <- c(seq, substr(genome, at, at + take - 1))
        left <- left - take
        if (left > 0) {
            ops <- c(ops, paste0(ex[k + 1, 1] - ex[k, 2] - 1, "N"))
            k <- k + 1; at <- ex[k, 1]
        }
    }
    list(pos = pos, cigar = paste(ops, collapse = ""), seq = paste(seq, collapse = ""))
}

revcomp <- function(s) paste(rev(strsplit(chartr("ACGTacgt", "TGCAtgca", s), "")[[1]]), collapse = "")

bam_dir <- file.path(opt$outdir, "bam")
dir.create(bam_dir, recursive = TRUE, showWarnings = FALSE)
ids <- sprintf("SAMPLE_%03d", seq_len(n))
for (i in seq_len(n)) {
    which_iso <- sample(names(isoforms), lib[i], replace = TRUE, prob = props[i, ])
    recs <- vapply(seq_len(lib[i]), function(r) {
        rd <- read_from(isoforms[[which_iso[r]]], opt$read_length)
        rev <- runif(1) < 0.5                    # unstranded library
        paste(sprintf("%s_r%05d", ids[i], r), if (rev) 16L else 0L, chrom, rd$pos, 255L,
              rd$cigar, "*", 0L, 0L, rd$seq, "*", "NH:i:1", sep = "\t")
    }, character(1))
    sam <- tempfile(fileext = ".sam")
    writeLines(c("@HD\tVN:1.6\tSO:unsorted",
                 sprintf("@SQ\tSN:%s\tLN:%d", chrom, nchar(genome)),
                 sprintf("@RG\tID:%s\tSM:%s", ids[i], ids[i]),
                 paste0(recs, "\tRG:Z:", ids[i])), sam)
    bam <- file.path(bam_dir, paste0(ids[i], ".bam"))
    # --no-PG keeps the build machine's paths out of the committed header
    if (system2("samtools", c("sort", "--no-PG", "-o", bam, sam)) != 0) stop("samtools sort failed")
    if (system2("samtools", c("index", bam)) != 0) stop("samtools index failed")
    unlink(sam)
}

# ---- manifest, gene list, and the simulated truth the tests check against
writeLines(c("sample_id\tstrand\tcoord_bam_list",
             sprintf("%s\tstrand_missing\tbam/%s.bam", ids, ids)),
           file.path(opt$outdir, "protocol_example.sample_list.txt"))
# AP000547.3 carries the reads; AP000547.4, just upstream, has none and so exercises the
# coverage filter of the count-matrix step.
writeLines(c("#chr\tstart\tend\tgene_id\tgene_name\tstrand",
             "chr22\t16592894\t16600110\tENSG00000287285\tAP000547.4\t+",
             "chr22\t16601910\t16615111\tENSG00000283633\tAP000547.3\t+"),
           file.path(opt$outdir, "protocol_example.genes.bed"))
truth <- data.frame(sample_id = ids, group = group, reads = lib, round(props, 4))
write.table(truth, file.path(opt$outdir, "simulated_isoform_proportions.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("wrote", n, "BAMs to", bam_dir, "\n")
