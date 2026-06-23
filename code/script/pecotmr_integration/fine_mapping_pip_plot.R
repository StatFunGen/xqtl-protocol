#!/usr/bin/env Rscript
# fine_mapping_pip_plot.R
#
# Render a per-region PIP landscape plot from a QtlFineMappingResult or
# GwasFineMappingResult RDS (output of fine_mapping.ipynb /
# multivariate_fine_mapping.ipynb / functional_fine_mapping.ipynb /
# gwas_rss_fine_mapping.ipynb). One PNG per input RDS, one facet panel
# per row in the FMR.
#
# Inputs:
#   --input    Path to a FineMappingResult RDS
#   --output   Output PNG path
#   --width    PNG width (inches). Default 9.
#   --height   Per-panel PNG height (inches). Default 3.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(ggplot2)
})

parser <- arg_parser("Render a per-region PIP plot from a FineMappingResult RDS")
parser <- add_argument(parser, "--input",
                       help = "Path to a FineMappingResult RDS",
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
if (!methods::is(fmr, "FineMappingResultBase"))
  stop("--input must be a FineMappingResultBase subclass (got '",
       class(fmr)[[1L]], "').")

write_empty <- function(msg) {
  message(msg, "; writing an empty plot to ", argv$output)
  dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
  png(argv$output, width = argv$width * 100, height = argv$height * 100)
  plot.new(); title(main = msg)
  dev.off()
}

if (nrow(fmr) == 0L) {
  write_empty("Empty FineMappingResult")
  quit(save = "no")
}

# Build a long tidy table of (panel-id, variant_id, pip) using the S4
# accessors. The panel label varies by FMR shape: GWAS keeps region_id,
# QTL uses (context, trait).
isGwas <- methods::is(fmr, "GwasFineMappingResult")
panels <- lapply(seq_len(nrow(fmr)), function(i) {
  entry <- fmr$entry[[i]]
  pip   <- as.numeric(getPip(entry))
  ids   <- as.character(getVariantIds(entry))
  if (length(pip) == 0L || length(ids) == 0L) return(NULL)
  panel <- if (isGwas) {
    sprintf("%s | %s | %s",
            as.character(fmr$study)[[i]],
            as.character(fmr$method)[[i]],
            as.character(fmr$region_id)[[i]])
  } else {
    sprintf("%s | %s | %s | %s",
            as.character(fmr$study)[[i]],
            as.character(fmr$context)[[i]],
            as.character(fmr$trait)[[i]],
            as.character(fmr$method)[[i]])
  }
  data.frame(panel = panel, variant_id = ids, pip = pip,
             stringsAsFactors = FALSE)
})
panels <- panels[!vapply(panels, is.null, logical(1))]
if (length(panels) == 0L) {
  write_empty("No usable PIP vectors on any entry")
  quit(save = "no")
}
df <- do.call(rbind, panels)

# Best-effort variant-id → position decode; falls back to row index.
pos <- suppressWarnings({
  m <- regmatches(df$variant_id,
                  regexec("^[^:_]+[:_]([0-9]+)", df$variant_id))
  vapply(m, function(x) if (length(x) >= 2L) as.numeric(x[[2L]]) else NA_real_,
         numeric(1L))
})
df$pos <- ifelse(is.na(pos), seq_len(nrow(df)), pos)

g <- ggplot(df, aes(x = pos, y = pip)) +
  geom_point(alpha = 0.6, size = 0.8) +
  facet_wrap(~ panel, ncol = 1, scales = "free_x") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Variant position", y = "PIP",
       title = paste0(class(fmr)[[1L]], ": ", basename(argv$input))) +
  theme_minimal(base_size = 10)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
ggsave(argv$output, g,
       width  = argv$width,
       height = argv$height * length(unique(df$panel)),
       dpi    = 150)
cat(sprintf("Wrote PIP plot for %d panel(s) (%d variants) to %s\n",
            length(unique(df$panel)), nrow(df), argv$output))
