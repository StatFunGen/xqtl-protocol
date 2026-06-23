#!/usr/bin/env Rscript
# fine_mapping_upset.R
#
# Render an UpSet plot of credible-set variant overlap across one or more
# FineMappingResult RDSes. Each (study, context, trait, method, region_id)
# tuple becomes one set; the set members are the variant IDs in the union
# of that tuple's credible sets. Useful for visualizing which signals are
# shared across studies / contexts / traits.
#
# Inputs:
#   --input <RDS> [<RDS> ...]  One or more FineMappingResult RDS paths.
#   --output <PNG>             Output PNG path.
#   --max-sets                 Cap on the number of sets to render (UpSet
#                              gets noisy past ~10). Default 20. Sets are
#                              ranked by their CS size (descending) and
#                              the top --max-sets are kept.
#   --width / --height         PNG dimensions in inches. Defaults 12 x 6.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("UpSet plot of credible-set variant overlap across FineMappingResult RDSes")
parser <- add_argument(parser, "--input",
                       help = "One or more FineMappingResult RDS paths",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--output",
                       help = "Output PNG path", type = "character")
parser <- add_argument(parser, "--max-sets",
                       help = "Maximum number of sets to render",
                       type = "integer", default = 20L)
parser <- add_argument(parser, "--width",
                       help = "PNG width (inches)",
                       type = "numeric", default = 12)
parser <- add_argument(parser, "--height",
                       help = "PNG height (inches)",
                       type = "numeric", default = 6)
argv <- parse_args(parser)

if (!requireNamespace("UpSetR", quietly = TRUE))
  stop("fine_mapping_upset.R requires the UpSetR package: ",
       "install.packages('UpSetR')")

inputs <- as.character(argv$input)
if (length(inputs) == 0L)
  stop("--input requires at least one RDS path.")

# Walk every (FMR-row x CS) combination and build a named list of
# {set-label -> variant_ids} that UpSetR consumes via fromList().
sets <- list()
for (path in inputs) {
  fmr <- readRDS(path)
  if (!methods::is(fmr, "FineMappingResultBase")) {
    warning("Skipping non-FineMappingResult input: ", path)
    next
  }
  isGwas <- methods::is(fmr, "GwasFineMappingResult")
  for (i in seq_len(nrow(fmr))) {
    entry <- fmr$entry[[i]]
    cs_df <- tryCatch(as.data.frame(getCs(entry)),
                      error = function(e) NULL)
    if (is.null(cs_df) || nrow(cs_df) == 0L) next
    if (!"variant_id" %in% names(cs_df)) next
    label <- if (isGwas) {
      sprintf("%s|%s|%s",
              as.character(fmr$study)[[i]],
              as.character(fmr$method)[[i]],
              as.character(fmr$region_id)[[i]])
    } else {
      sprintf("%s|%s|%s|%s",
              as.character(fmr$study)[[i]],
              as.character(fmr$context)[[i]],
              as.character(fmr$trait)[[i]],
              as.character(fmr$method)[[i]])
    }
    sets[[label]] <- unique(as.character(cs_df$variant_id))
  }
}

if (length(sets) == 0L) {
  message("No credible sets found across any input; writing an empty plot.")
  dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
  png(argv$output, width = argv$width * 100, height = argv$height * 100)
  plot.new(); title(main = "No credible sets in inputs")
  dev.off()
  quit(save = "no")
}

# Rank sets by size descending, cap at --max-sets to keep the plot readable.
sizes <- vapply(sets, length, integer(1L))
keep  <- order(sizes, decreasing = TRUE)[seq_len(min(argv$max_sets, length(sets)))]
sets  <- sets[keep]
if (length(sets) < length(sizes))
  message("Rendering top ", argv$max_sets, " set(s) by size; ",
          length(sizes) - length(sets), " smaller set(s) dropped.")

binary <- UpSetR::fromList(sets)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
png(argv$output, width = argv$width * 100, height = argv$height * 100)
print(UpSetR::upset(
  binary,
  nsets   = length(sets),
  order.by = "freq",
  mainbar.y.label = "CS-variant intersection size",
  sets.x.label    = "CS variants per (study, context, trait, method)"))
dev.off()
cat(sprintf("Wrote UpSet plot of %d set(s) to %s\n",
            length(sets), argv$output))
