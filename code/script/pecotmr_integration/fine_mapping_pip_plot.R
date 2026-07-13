#!/usr/bin/env Rscript
# fine_mapping_pip_plot.R
#
# Render the per-region PIP landscape of a QtlFineMappingResult /
# GwasFineMappingResult RDS. Points are coloured by credible set and shaped by
# whether the variant is shared (in a CS) across >1 panel -- the same encoding
# as the legacy susie_pip_landscape_plot. When --annot-tibble is supplied (an
# Annotatr builtin-annotation TSV: seqnames/start/end/type/symbol), a
# regulatory-element track and a gene track are stacked under the PIP panel via
# cowplot, reproducing the legacy composite figure. Without it, only the PIP
# landscape is drawn (pecotmr carries no annotation resource).
#
# Inputs:
#   --input          FineMappingResult RDS.
#   --output         Output plot path (.pdf / .png; device inferred).
#   --annot-tibble   Optional Annotatr annotation TSV for the overlay tracks.
#   --width          Plot width (inches). Default 12.
#   --height         Per-PIP-panel height (inches). Default 3.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(ggplot2)
})

parser <- arg_parser("Render the PIP landscape of a FineMappingResult RDS")
parser <- add_argument(parser, "--input", help = "FineMappingResult RDS", type = "character")
parser <- add_argument(parser, "--output", help = "Output plot path (.pdf/.png)", type = "character")
parser <- add_argument(parser, "--annot-tibble",
                       help = "Optional Annotatr annotation TSV for the overlay tracks",
                       type = "character", default = NA)
parser <- add_argument(parser, "--width", help = "Plot width in inches", type = "numeric", default = 12)
parser <- add_argument(parser, "--height", help = "Per-PIP-panel height in inches", type = "numeric", default = 3)
argv <- parse_args(parser)

fmr <- readRDS(argv$input)
if (!methods::is(fmr, "FineMappingResultBase"))
  stop("--input must be a FineMappingResultBase subclass (got '", class(fmr)[[1L]], "').")

# Discrete palette matching the legacy plot (CS index -> colour).
palette <- c("black", "dodgerblue2", "green4", "#6A3D9A", "#FF7F00", "gold1",
             "skyblue2", "#FB9A99", "palegreen2", "#CAB2D6", "#FDBF6F", "gray70",
             "khaki2", "maroon", "orchid1", "deeppink1", "blue1", "steelblue4",
             "darkturquoise", "green1", "yellow4", "yellow3", "darkorange4",
             "brown", "navyblue", "#FF0000", "darkgreen", "#FFFF00", "purple")

writeStub <- function(msg) {
  message(msg, "; writing an empty plot to ", argv$output)
  dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
  ggsave(argv$output, ggplot() + theme_void() + labs(title = msg),
         width = argv$width, height = argv$height)
}

if (nrow(fmr) == 0L) { writeStub("Empty FineMappingResult"); quit(save = "no") }

# getTopLoci(signalCutoff = 0): every fitted variant with pos, pip, CS membership
# (cs_95 = "<method>_<index>"), and the row-identity columns -> one facet panel.
tl <- tryCatch(as.data.frame(getTopLoci(fmr, signalCutoff = 0)), error = function(e) NULL)
if (is.null(tl) || nrow(tl) == 0L) { writeStub("No usable PIP vectors"); quit(save = "no") }

isGwas <- methods::is(fmr, "GwasFineMappingResult")
panel <- if (isGwas) {
  sprintf("%s | %s | %s", tl$study, tl$method, tl$region_id)
} else {
  sprintf("%s | %s | %s", tl$context, tl$trait, tl$method)
}
pos <- suppressWarnings(as.numeric(tl$pos)); pos[is.na(pos)] <- seq_len(nrow(tl))[is.na(pos)]
csIdx <- suppressWarnings(as.integer(sub(".*_", "", as.character(tl$cs_95))))
csIdx[is.na(csIdx)] <- 0L
df <- data.frame(panel = panel, variant_id = as.character(tl$variant_id),
                 pip = as.numeric(tl$pip), pos = pos, CS = factor(csIdx),
                 stringsAsFactors = FALSE)
# Shared: a variant that sits in a credible set (CS != 0) in more than one panel.
inCs <- df[df$CS != "0", , drop = FALSE]
sharedIds <- unique(inCs$variant_id[duplicated(inCs$variant_id)])
df$Shared <- df$variant_id %in% sharedIds

plotRange <- c(min(df$pos), max(df$pos))
chrom <- sub(":.*", "", df$variant_id[[1L]])

pipPlot <- ggplot(df, aes(x = pos, y = pip, colour = CS, shape = Shared)) +
  geom_point(size = 3, alpha = 0.8) +
  facet_grid(panel ~ .) +
  scale_colour_manual("CS", values = palette) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "PIP", title = paste0("Fine-mapping overview: ", basename(argv$input))) +
  theme_minimal(base_size = 14) +
  theme(strip.text.y.right = element_text(angle = 0))

panels <- list(pipPlot)
relHeights <- c(8)

# Optional annotation overlay (regulatory-element + gene tracks) when the
# Annotatr tibble is supplied. Columns: seqnames, start, end, type, symbol.
if (!is.na(argv$annot_tibble) && file.exists(argv$annot_tibble)) {
  annot <- tryCatch(read.delim(argv$annot_tibble, stringsAsFactors = FALSE),
                    error = function(e) NULL)
  if (!is.null(annot) && all(c("seqnames", "start", "end", "type") %in% names(annot))) {
    annot <- annot[annot$seqnames == chrom & annot$start > plotRange[1] &
                     annot$end < plotRange[2], , drop = FALSE]
    reg <- annot[!annot$type %in% c("hg38_genes_introns", "hg38_genes_1to5kb"), , drop = FALSE]
    if (nrow(reg) > 0L) {
      panels <- c(panels, list(
        ggplot(reg) +
          geom_segment(aes(x = start, xend = end, y = "Regulatory", yend = "Regulatory",
                           colour = type), linewidth = 6) +
          labs(x = "", y = "") + xlim(plotRange) +
          theme_minimal(base_size = 12) + theme(axis.text.x = element_blank())))
      relHeights <- c(relHeights, 1)
    }
    genes <- annot[annot$type == "hg38_genes_1to5kb" & !is.na(annot$symbol), , drop = FALSE]
    if (nrow(genes) > 0L) {
      g <- do.call(rbind, lapply(split(genes, genes$symbol), function(s)
        data.frame(symbol = s$symbol[[1L]], start = min(s$start), end = max(s$end))))
      panels <- c(panels, list(
        ggplot(g) +
          geom_segment(aes(x = start, xend = end, y = "Gene", yend = "Gene", colour = symbol),
                       linewidth = 6) +
          geom_label(aes(x = (start + end) / 2, y = "Gene", label = symbol), size = 3) +
          labs(x = "POS", y = "") + xlim(plotRange) +
          theme_minimal(base_size = 12) + theme(legend.position = "none")))
      relHeights <- c(relHeights, 1)
    }
  }
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
nPanel <- length(unique(df$panel))
if (length(panels) == 1L) {
  ggsave(argv$output, pipPlot, width = argv$width, height = argv$height * nPanel, limitsize = FALSE)
} else if (requireNamespace("cowplot", quietly = TRUE)) {
  grid <- cowplot::plot_grid(plotlist = panels, ncol = 1, align = "v",
                             axis = "tlbr", rel_heights = relHeights)
  ggsave(argv$output, grid, width = argv$width,
         height = argv$height * nPanel + 2 * (length(panels) - 1), limitsize = FALSE)
} else {
  message("cowplot not installed; writing the PIP panel only.")
  ggsave(argv$output, pipPlot, width = argv$width, height = argv$height * nPanel, limitsize = FALSE)
}
cat(sprintf("Wrote PIP landscape (%d panel(s), %d annotation track(s)) to %s\n",
            nPanel, length(panels) - 1L, argv$output))
