#!/usr/bin/env Rscript
# fine_mapping_upset.R
#
# Render an UpSet plot of credible-set variant overlap across one or more
# FineMappingResult RDSes. Each (study, context, trait, method, blockId)
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

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "pecotmr_aliases.R"))

if (!requireNamespace("UpSetR", quietly = TRUE))
  stop("fine_mapping_upset.R requires the UpSetR package: ",
       "install.packages('UpSetR')")

inputs <- as.character(argv$input)
if (length(inputs) == 0L)
  stop("--input requires at least one RDS path.")

# Build a named list of {set-label -> variant_ids} for UpSetR::fromList().
# credibleSets(fmr) aggregates every entry's credible sets into one table already
# tagged with the row identity (study/context/trait/blockId/method), so each
# per-tuple set is a split() on the label -- no per-entry loop.
sets <- list()
for (path in inputs) {
  fmr <- readRDS(path)
  if (!methods::is(fmr, "FineMappingResultBase")) {
    warning("Skipping non-FineMappingResult input: ", path)
    next
  }
  # Unguarded: an FMR with no credible sets yields a 0-row table, which the
  # next line skips; an accessor error is a real break and must surface.
  cs <- as.data.frame(credibleSets(fmr))
  if (nrow(cs) == 0L || !"variant_id" %in% names(cs)) next
  label <- if (methods::is(fmr, "GwasFineMappingResult")) {
    paste(cs$study, cs$method, cs$blockId, sep = "|")
  } else {
    paste(cs$study, cs$context, cs$trait, cs$method, sep = "|")
  }
  perLabel <- split(as.character(cs$variant_id), label)
  for (lab in names(perLabel))
    sets[[lab]] <- unique(c(sets[[lab]], perLabel[[lab]]))
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
