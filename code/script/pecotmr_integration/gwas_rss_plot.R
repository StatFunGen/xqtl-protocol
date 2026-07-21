#!/usr/bin/env Rscript
# gwas_rss_plot.R
#
# Render a per-region PIP plot from a pecotmr::GwasFineMappingResult RDS
# (output of fine_mapping.R --gwas-sumstats). One PNG per input RDS;
# each row in the FMR produces one panel (study × method × region_id).
# Replaces the legacy pipeline/rss_analysis.ipynb [univariate_plot] step.
#
# Inputs:
#   --input    Path to a GwasFineMappingResult RDS
#   --output   Output PNG path
#   --width    PNG width (inches). Default 9.
#   --height   PNG height (inches). Default 3 per panel.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(ggplot2)
})

parser <- arg_parser("Render a PIP plot from a GwasFineMappingResult RDS")
parser <- add_argument(parser, "--input",
                       help = "Path to a GwasFineMappingResult RDS",
                       type = "character")
parser <- add_argument(parser, "--output",
                       help = "Output PNG path", type = "character")
parser <- add_argument(parser, "--width",
                       help = "PNG width in inches",
                       type = "numeric", default = 9)
parser <- add_argument(parser, "--height",
                       help = "Per-panel PNG height in inches",
                       type = "numeric", default = 3)
argv <- parse_args(parser)

write_empty <- function(msg) {
  message(msg, "; writing an empty plot to ", argv$output)
  dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
  ggsave(argv$output, ggplot() + theme_void() + labs(title = msg),
         width = argv$width, height = argv$height, dpi = 150)
}

fmr <- readRDS(argv$input)
if (!methods::is(fmr, "GwasFineMappingResult"))
  stop("--input must be a GwasFineMappingResult (got '",
       class(fmr)[[1L]], "').")
if (nrow(fmr) == 0L) {
  write_empty("No fine-mapping rows in input")
  quit(save = "no")
}

# One aggregated PIP table for the collection. getTopLoci(signalCutoff = 0)
# returns every variant with its decoded `pos` and the row-identity columns
# (study / method / region_id), so the panel label and position come straight
# off the table -- one panel per (study, method, region_id).
tl <- getTopLoci(fmr, signalCutoff = 0)
if (is.null(tl) || nrow(tl) == 0L) {
  write_empty("No PIP vectors in input")
  quit(save = "no")
}
pos <- suppressWarnings(as.numeric(tl$pos))
pos[is.na(pos)] <- seq_len(nrow(tl))[is.na(pos)]
df <- data.frame(
  variant_id = as.character(tl$variant_id),
  pip        = as.numeric(tl$pip),
  pos        = pos,
  panel      = paste(tl$study, tl$method, tl$region_id, sep = " | "),
  stringsAsFactors = FALSE)

g <- ggplot(df, aes(x = pos, y = pip)) +
  geom_point(alpha = 0.6, size = 0.8) +
  facet_wrap(~ panel, ncol = 1, scales = "free_x") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Variant position", y = "PIP",
       title = paste0("GWAS RSS fine-mapping: ", basename(argv$input))) +
  theme_minimal(base_size = 10)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
ggsave(argv$output, g,
       width  = argv$width,
       height = argv$height * length(unique(df$panel)),
       dpi    = 150)
cat(sprintf("Wrote PIP plot for %d panel(s) (%d total variants) to %s\n",
            length(unique(df$panel)), nrow(df), argv$output))
