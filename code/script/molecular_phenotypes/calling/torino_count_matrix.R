#!/usr/bin/env Rscript
# ============================================================
# torino_count_matrix.R
# Per-gene read-depth matrices, the input Torino factorizes, for the torino_2 step
# of isoform_calling_torino.ipynb. Replaces prepare_count_tables.py from the
# Torino paper repository (cfbuenabadn/torino_paper, code/scripts).
#
# For each gene in --genes the window [start - flank, end + flank) is read from
# every sample's coverage bedGraph and written as one CSV, samples as rows and
# bases as columns: a Sample_ID column, then one column per base named
# <chrom>:<position>, 1-based. This is the layout torino::run_torino_gene reads.
#
# Coverage files are found by the names torino_coverage.sh gives them:
#   <coverage-dir>/<sample_id>.coverage.bedgraph.gz              strand_missing
#   <coverage-dir>/<sample_id>.{plus,minus}.coverage.bedgraph.gz rf, fr
# A stranded sample contributes the file matching the gene's strand. Unstranded
# libraries cannot separate genes that overlap on opposite strands; the Torino
# paper drops such regions for GTEx, and this script leaves that to the gene list.
#
# A gene is written only if at least --min-samples samples have a summed depth
# (the sum over the window of per-base depth, the quantity Torino's own
# min_counts filters on) of at least --min-counts. Genes failing the filter are
# listed, with the reason, in --skipped.
#
# --chunk i --n-chunks n processes the i-th of n contiguous slices of --genes, so
# a cohort-wide run can be split across jobs.
#
# Writes <output-dir>/<gene_id>.csv.gz for each kept gene, --output-list (one CSV
# path per line) and --skipped (gene_id, gene_name, reason; header always written).
# ============================================================

suppressPackageStartupMessages({
    library(argparser)
    library(data.table)
})

parser <- arg_parser("Per-gene depth matrices for Torino")
parser <- add_argument(parser, "--samples", type = "character",
                       help = "sample manifest with sample_id and strand columns")
parser <- add_argument(parser, "--coverage-dir", type = "character",
                       help = "directory holding torino_coverage.sh output")
parser <- add_argument(parser, "--genes", type = "character",
                       help = "gene BED: chrom, start (0-based), end, gene_id, gene_name, strand")
parser <- add_argument(parser, "--output-dir", type = "character", help = "directory for the per-gene CSVs")
parser <- add_argument(parser, "--output-list", type = "character", help = "file listing the CSVs written")
parser <- add_argument(parser, "--skipped", type = "character", help = "file listing genes not written")
parser <- add_argument(parser, "--flank", type = "integer", default = 50L,
                       help = "bases added on each side of the gene")
parser <- add_argument(parser, "--min-counts", type = "integer", default = 100L,
                       help = "summed depth a sample needs over the window to count towards --min-samples")
parser <- add_argument(parser, "--min-samples", type = "integer", default = 10L,
                       help = "samples that must reach --min-counts for the gene to be written")
parser <- add_argument(parser, "--chunk", type = "integer", default = 1L, help = "slice of --genes to process")
parser <- add_argument(parser, "--n-chunks", type = "integer", default = 1L, help = "number of slices")
argv <- parse_args(parser)
for (a in c("samples", "coverage_dir", "genes", "output_dir", "output_list", "skipped"))
    if (is.na(argv[[a]])) stop("--", gsub("_", "-", a), " is required")

die <- function(...) { message("ERROR: ", ...); quit(status = 1) }
if (argv$n_chunks < 1 || argv$chunk < 1 || argv$chunk > argv$n_chunks)
    die("--chunk must be between 1 and --n-chunks (got ", argv$chunk, " of ", argv$n_chunks, ")")
if (argv$flank < 0) die("--flank must be zero or more")
if (Sys.which("tabix") == "") die("tabix is not on PATH")

# ---- samples and their coverage files
manifest <- fread(argv$samples, sep = "\t", colClasses = "character")
for (col in c("sample_id", "strand"))
    if (!col %in% names(manifest)) die(argv$samples, " has no '", col, "' column")
if (anyDuplicated(manifest$sample_id)) die("duplicated sample_id in ", argv$samples)
# the regtools codes other splicing manifests use are accepted as aliases
alias <- c(XS = "strand_missing", RF = "rf", FR = "fr")
manifest[strand %in% names(alias), strand := alias[strand]]
bad <- setdiff(unique(manifest$strand), c("strand_missing", "rf", "fr"))
if (length(bad)) die("unknown strand value(s) ", paste(bad, collapse = ", "), " in ", argv$samples)

cov_file <- function(id, strand, gene_strand) {
    tag <- if (strand == "strand_missing") "" else if (gene_strand == "-") ".minus" else ".plus"
    file.path(argv$coverage_dir, paste0(id, tag, ".coverage.bedgraph.gz"))
}
needed <- unlist(lapply(seq_len(nrow(manifest)), function(i)
    if (manifest$strand[i] == "strand_missing") cov_file(manifest$sample_id[i], "strand_missing", "+")
    else c(cov_file(manifest$sample_id[i], "rf", "+"), cov_file(manifest$sample_id[i], "rf", "-"))))
missing <- needed[!file.exists(needed) | !file.exists(paste0(needed, ".tbi"))]
if (length(missing))
    die(length(missing), " coverage file(s) or index(es) missing, e.g. ", missing[1])

# ---- genes, this chunk's slice
gene_lines <- readLines(gzfile(argv$genes))                 # plain or gzipped
gene_lines <- gene_lines[nzchar(gene_lines) & !startsWith(gene_lines, "#")]
if (!length(gene_lines)) die(argv$genes, " lists no genes")
genes <- fread(text = gene_lines, sep = "\t", header = FALSE,
               colClasses = list(character = c(1, 4, 5, 6), integer = c(2, 3)))
if (ncol(genes) < 6) die(argv$genes, " needs six columns: chrom, start, end, gene_id, gene_name, strand")
setnames(genes, 1:6, c("chrom", "start", "end", "gene_id", "gene_name", "strand"))
if (anyDuplicated(genes$gene_id)) die("duplicated gene_id in ", argv$genes)
if (!all(genes$strand %in% c("+", "-"))) die("gene strand must be + or - in ", argv$genes)
per_chunk <- ceiling(nrow(genes) / argv$n_chunks)
genes <- genes[(seq_len(.N) - 1L) %/% per_chunk + 1L == argv$chunk]

# Reconcile "chr22" and "22" naming against what the coverage files hold.
seqs <- system2("tabix", c("-l", shQuote(needed[1])), stdout = TRUE)
genes[, query_chrom := chrom]
genes[!chrom %in% seqs & paste0("chr", chrom) %in% seqs, query_chrom := paste0("chr", chrom)]
genes[!chrom %in% seqs & sub("^chr", "", chrom) %in% seqs, query_chrom := sub("^chr", "", chrom)]
if (nrow(genes) && !any(genes$query_chrom %in% seqs))
    die("no gene chromosome of this chunk (", paste(head(unique(genes$chrom), 3), collapse = ", "),
        ") is in the coverage files, which hold ", paste(head(seqs, 3), collapse = ", "), " ...")

# ---- depth of one sample over [ws, we), 0-based half-open
window_depth <- function(file, chrom, ws, we) {
    depth <- integer(we - ws)
    region <- sprintf("%s:%d-%d", chrom, ws + 1L, we)
    hits <- system2("tabix", c(shQuote(file), shQuote(region)), stdout = TRUE)
    if (!length(hits)) return(depth)
    bg <- fread(text = hits, header = FALSE, col.names = c("chrom", "s", "e", "d"))
    for (r in seq_len(nrow(bg))) {
        a <- max(bg$s[r], ws); b <- min(bg$e[r], we)
        if (b > a) depth[(a - ws + 1L):(b - ws)] <- as.integer(bg$d[r])
    }
    depth
}

dir.create(argv$output_dir, recursive = TRUE, showWarnings = FALSE)
written <- character(0)
skipped <- data.table(gene_id = character(0), gene_name = character(0), reason = character(0))
for (g in seq_len(nrow(genes))) {
    gn <- genes[g]
    if (!gn$query_chrom %in% seqs) {
        skipped <- rbind(skipped, data.table(gn$gene_id, gn$gene_name, "chromosome not in coverage files"), use.names = FALSE)
        next
    }
    ws <- max(0L, as.integer(gn$start) - argv$flank)
    we <- as.integer(gn$end) + argv$flank
    m <- vapply(seq_len(nrow(manifest)), function(i)
        window_depth(cov_file(manifest$sample_id[i], manifest$strand[i], gn$strand), gn$query_chrom, ws, we),
        integer(we - ws))
    m <- matrix(m, nrow = we - ws)                           # bases x samples, even for one sample
    passing <- sum(colSums(m) >= argv$min_counts)
    if (passing < argv$min_samples) {
        skipped <- rbind(skipped, data.table(gn$gene_id, gn$gene_name,
            sprintf("%d sample(s) with depth >= %d, %d needed", passing, argv$min_counts, argv$min_samples)),
            use.names = FALSE)
        next
    }
    dt <- data.table(Sample_ID = manifest$sample_id, t(m))
    setnames(dt, c("Sample_ID", paste0(gn$chrom, ":", seq.int(ws + 1L, we))))
    out <- file.path(argv$output_dir, paste0(gn$gene_id, ".csv.gz"))
    fwrite(dt, out, compress = "gzip")
    written <- c(written, out)
}

writeLines(written, argv$output_list)
fwrite(skipped, argv$skipped, sep = "\t")
message(sprintf("chunk %d/%d: %d gene(s) written, %d skipped", argv$chunk, argv$n_chunks,
                length(written), nrow(skipped)))
