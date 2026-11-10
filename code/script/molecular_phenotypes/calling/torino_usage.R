#!/usr/bin/env Rscript
# ============================================================
# torino_usage.R
# Collects the per-gene Torino factorizations into phenotype tables, for the
# torino_4 step of isoform_calling_torino.ipynb. Replaces collect_loadings.R and
# QQNorm_factors.py from the Torino paper repository (cfbuenabadn/torino_paper,
# code/scripts).
#
# Each gene contributes one row per coverage component: its usage (the EL loading
# Torino estimates for each sample) across samples, with NA for samples Torino
# dropped below its depth threshold. Rows are keyed <gene_id>:factor_<k> and
# placed at the gene's coordinates from --genes.
#
# Normalization follows QQNorm_factors.py. A component is kept if at most
# --max-missing of samples are NA and its 90th-percentile usage is at least
# --min-usage-q90; NAs are filled with the component's mean, each component is
# centred and scaled across samples, and each sample is then quantile-normalized
# to a standard normal across components.
#
# Writes, with <p> = --output-prefix:
#   <p>.torino_usage.bed.gz             raw usage, every component
#   <p>.torino_usage.qqnorm.bed.gz      the normalized phenotype, kept components
#   <p>.torino_usage.phenotype_group.txt  component -> gene, for the kept components
# The two BEDs are coordinate-sorted, bgzipped and tabix-indexed, with header
# #chr start end ID <sample_id>..., the layout the association_scan modules read.
# ============================================================

suppressPackageStartupMessages({
    library(argparser)
    library(data.table)
})

parser <- arg_parser("Collect Torino usages into phenotype BEDs")
parser <- add_argument(parser, "--factorizations", type = "character", nargs = Inf,
                       help = "RDS files from torino_factorize.R")
parser <- add_argument(parser, "--genes", type = "character", help = "gene BED used to build the count matrices")
parser <- add_argument(parser, "--samples", type = "character", help = "sample manifest; fixes the column order")
parser <- add_argument(parser, "--output-prefix", type = "character", help = "<dir>/<prefix> for the outputs")
parser <- add_argument(parser, "--max-missing", type = "numeric", default = 0.1,
                       help = "largest fraction of NA samples a kept component may have")
parser <- add_argument(parser, "--min-usage-q90", type = "numeric", default = 0.1,
                       help = "smallest 90th-percentile usage a kept component may have")
argv <- parse_args(parser)
for (a in c("genes", "samples", "output_prefix"))
    if (is.na(argv[[a]])) stop("--", gsub("_", "-", a), " is required")
rds <- argv$factorizations
if (length(rds) == 0 || all(is.na(rds))) stop("--factorizations is required")

die <- function(...) { message("ERROR: ", ...); quit(status = 1) }
for (tool in c("bgzip", "tabix")) if (Sys.which(tool) == "") die(tool, " is not on PATH")

samples <- fread(argv$samples, sep = "\t", colClasses = "character")$sample_id
if (is.null(samples)) die(argv$samples, " has no 'sample_id' column")
gene_lines <- readLines(gzfile(argv$genes))
gene_lines <- gene_lines[nzchar(gene_lines) & !startsWith(gene_lines, "#")]
genes <- fread(text = gene_lines, sep = "\t", header = FALSE,
               colClasses = list(character = c(1, 4, 5, 6), integer = c(2, 3)))
setnames(genes, 1:6, c("chrom", "start", "end", "gene_id", "gene_name", "strand"))
setkey(genes, gene_id)

# ---- one block of rows per gene
rows <- list()
failed <- character(0)
for (f in rds) {
    fit <- readRDS(f)
    slot <- grep("^torino_K[0-9]+$", names(fit), value = TRUE)
    if (length(slot) != 1) die(f, " is not a torino_factorize.R output (no torino_K<K> element)")
    tor <- fit[[slot]]
    if (is.null(tor)) { failed <- c(failed, fit$gene); next }
    g <- genes[J(fit$gene), nomatch = NULL]
    if (nrow(g) != 1) die("gene ", fit$gene, " from ", f, " is not in ", argv$genes)
    el <- tor$EL
    usage <- matrix(NA_real_, ncol(el), length(samples), dimnames = list(NULL, samples))
    hit <- intersect(rownames(el), samples)
    usage[, hit] <- t(el[hit, , drop = FALSE])
    rows[[length(rows) + 1]] <- data.table(`#chr` = g$chrom, start = g$start, end = g$end,
                                           ID = paste0(g$gene_id, ":", colnames(el)), gene = g$gene_id,
                                           usage)
}
if (length(failed))
    message(length(failed), " gene(s) skipped because Torino returned no factorization: ",
            paste(head(failed, 10), collapse = ", "), if (length(failed) > 10) " ...")
if (!length(rows)) die("no factorization in --factorizations holds a result")
usage <- rbindlist(rows)
setorderv(usage, c("#chr", "start", "end", "ID"))

# ---- normalization (QQNorm_factors.py)
qqnorm_col <- function(x) {
    n <- length(x); a <- if (n <= 10) 3 / 8 else 0.5
    qnorm((rank(x) - a) / (n + 1 - 2 * a))
}
u <- as.matrix(usage[, ..samples])
missing <- rowMeans(is.na(u))
q90 <- apply(u, 1, quantile, probs = 0.9, na.rm = TRUE, names = FALSE)
keep <- missing <= argv$max_missing & !is.na(q90) & q90 >= argv$min_usage_q90
message(sum(keep), " of ", nrow(u), " component(s) kept for the normalized table")
if (!any(keep)) die("no component passes --max-missing ", argv$max_missing,
                    " and --min-usage-q90 ", argv$min_usage_q90)
k <- u[keep, , drop = FALSE]
for (i in seq_len(nrow(k))) k[i, is.na(k[i, ])] <- mean(k[i, ], na.rm = TRUE)
k <- t(scale(t(k)))
k[is.na(k)] <- 0                                  # a component with no variance scales to NaN
k <- apply(k, 2, qqnorm_col)
if (!is.matrix(k)) k <- matrix(k, nrow = 1, dimnames = list(NULL, samples))
normalized <- cbind(usage[keep, .(`#chr`, start, end, ID)], as.data.table(k))

# ---- write
write_bed <- function(dt, path) {
    plain <- sub("\\.gz$", "", path)
    fwrite(dt, plain, sep = "\t", quote = FALSE, na = "NA")
    if (system2("bgzip", c("-f", shQuote(plain))) != 0) die("bgzip failed on ", plain)
    if (system2("tabix", c("-f", "-p", "bed", shQuote(path))) != 0) die("tabix failed on ", path)
}
dir.create(dirname(argv$output_prefix), recursive = TRUE, showWarnings = FALSE)
p <- argv$output_prefix
write_bed(usage[, c("#chr", "start", "end", "ID", samples), with = FALSE], paste0(p, ".torino_usage.bed.gz"))
write_bed(normalized, paste0(p, ".torino_usage.qqnorm.bed.gz"))
fwrite(usage[keep, .(ID, gene)], paste0(p, ".torino_usage.phenotype_group.txt"),
       sep = "\t", col.names = FALSE)
