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

fmr <- readRDS(argv$input)
if (!methods::is(fmr, "GwasFineMappingResult"))
  stop("--input must be a GwasFineMappingResult (got '",
       class(fmr)[[1L]], "').")
if (nrow(fmr) == 0L) {
  message("Empty GwasFineMappingResult; writing an empty plot to ", argv$output)
  png(argv$output, width = argv$width * 100, height = argv$height * 100)
  plot.new(); title(main = "No fine-mapping rows in input")
  dev.off()
  quit(save = "no")
}

# Build a tidy long table of (study, method, region_id, variant_id, pip)
# straight from the S4 accessors; one panel per row of the FMR.
panels <- lapply(seq_len(nrow(fmr)), function(i) {
  entry  <- fmr$entry[[i]]
  pip    <- as.numeric(getPip(entry))
  ids    <- as.character(getVariantIds(entry))
  if (length(pip) == 0L || length(ids) == 0L) return(NULL)
  data.frame(
    study      = as.character(fmr$study)[[i]],
    method     = as.character(fmr$method)[[i]],
    region_id  = as.character(fmr$region_id)[[i]],
    variant_id = ids,
    pip        = pip,
    stringsAsFactors = FALSE)
})
panels <- panels[!vapply(panels, is.null, logical(1))]
if (length(panels) == 0L) {
  message("No usable PIP vectors on any entry; writing an empty plot to ",
          argv$output)
  png(argv$output, width = argv$width * 100, height = argv$height * 100)
  plot.new(); title(main = "No PIP vectors in input")
  dev.off()
  quit(save = "no")
}
df <- do.call(rbind, panels)

# Extract a numeric pos when variant_id looks like "chrN:pos:..." or
# "chrN_pos_..."; otherwise plot against the row index. This is a
# best-effort cosmetic decode, not a hard requirement.
pos <- suppressWarnings({
  m <- regmatches(df$variant_id,
                  regexec("^[^:_]+[:_]([0-9]+)", df$variant_id))
  vapply(m, function(x) if (length(x) >= 2L) as.numeric(x[[2L]]) else NA_real_,
         numeric(1L))
})
df$pos <- ifelse(is.na(pos), seq_len(nrow(df)), pos)
df$panel <- paste(df$study, df$method, df$region_id, sep = " | ")

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
