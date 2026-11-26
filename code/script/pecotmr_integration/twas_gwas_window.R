#!/usr/bin/env Rscript
# twas_gwas_window.R
#
# The window one twas_ctwas.ipynb [twas] job builds its GwasSumStats over: the
# job's analysis unit, widened to every variant of its genes' TWAS weights and
# fine-mapping results. causalInferencePipeline() refuses a gene whose weights
# or fine-mapping result reach past the GWAS window it is given, as those
# variants would lie in blocks never loaded; and a unit does not cover its
# genes by itself, since a region holds genes by TSS and a gene unit spans only
# the gene's xQTL-meta window.
#
# Inputs:
#   --gene-table  twas.R's gene table: columns twas_weights and, optionally,
#                 fine_mapping_result (empty, ".", or a missing file means
#                 none, as in twas.R)
#   --unit        The unit's window, chr:start-end
#
# Output: the window, chr:start-end, on stdout.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("GWAS window covering a twas job's weights and fine-mapping results")
parser <- add_argument(parser, "--gene-table", type = "character",
                       help = "twas.R gene table (twas_weights, fine_mapping_result)")
parser <- add_argument(parser, "--unit", type = "character",
                       help = "The analysis unit's window, chr:start-end")
argv <- parse_args(parser)

genes <- vroom::vroom(argv$gene_table, delim = "\t", col_types = vroom::cols(.default = "c"),
                      na = character(), progress = FALSE)
fm <- genes[["fine_mapping_result"]]
fm <- fm[nzchar(fm) & fm != "." & file.exists(fm)]
spans <- lapply(c(genes$twas_weights, fm),
                function(f) range(unlist(readRDS(f)), ignore.strand = TRUE))

unit <- GenomicRanges::GRanges(argv$unit)
window <- range(c(unit, unlist(GenomicRanges::GRangesList(spans))), ignore.strand = TRUE)
# Variants on another chromosome sit in no block of this one, so pecotmr does
# not check them against it.
window <- window[GenomicRanges::seqnames(window) == GenomicRanges::seqnames(unit)]
cat(sprintf("%s:%d-%d\n", GenomicRanges::seqnames(window),
            GenomicRanges::start(window), GenomicRanges::end(window)))
