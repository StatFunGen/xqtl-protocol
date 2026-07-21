#!/usr/bin/env Rscript
# mash_plot_prior.R
#
# Plot the MASH prior covariance matrices as sharing heatmaps -- the wrapper
# backing mixture_prior.ipynb's `plot_U` step. Reads a prior RDS (a
# list(U, w[, loglik]) as written by mash_prior.R) and renders one heatmap per
# weighted covariance component into a multi-panel PDF.
#
# Inputs:
#   --data          Prior RDS (list with $U covariance list + $w weights).
#   --name          Optional sub-element of the RDS to plot (dat[[name]]).
#   --max-comp      Max components to show (-1 = all above --tol). Default -1.
#   --to-cor        Convert covariances to correlations before plotting.
#   --tol           Weight threshold for "shown" components. Default 1E-6.
#   --remove-label  Replace condition names with t1..tn.
#   --output        Output PDF path.

suppressPackageStartupMessages({
  library(argparser)
  library(reshape2)
  library(ggplot2)
})

p <- arg_parser("Plot MASH prior covariance matrices as sharing heatmaps")
p <- add_argument(p, "--data", type = "character", help = "prior RDS (list(U, w))")
p <- add_argument(p, "--name", type = "character", default = "",
                  help = "optional sub-element of the RDS to plot")
p <- add_argument(p, "--max-comp", type = "integer", default = -1L,
                  help = "max components to show (-1 = all above --tol)")
p <- add_argument(p, "--to-cor", flag = TRUE,
                  help = "convert covariances to correlations")
p <- add_argument(p, "--tol", type = "numeric", default = 1e-6,
                  help = "weight threshold for shown components")
p <- add_argument(p, "--remove-label", flag = TRUE,
                  help = "replace condition names with t1..tn")
p <- add_argument(p, "--output", type = "character", help = "output PDF path")
argv <- parse_args(p)

plot_sharing <- function(X, col = "black", to_cor = FALSE, title = "",
                         remove_names = FALSE) {
  clrs <- colorRampPalette(rev(c("#D73027", "#FC8D59", "#FEE090", "#FFFFBF",
    "#E0F3F8", "#91BFDB", "#4575B4")))(128)
  lat <- if (to_cor) cov2cor(X) else X / max(diag(X))
  lat[lower.tri(lat)] <- NA
  n <- nrow(lat)
  if (remove_names) {
    colnames(lat) <- rownames(lat) <- paste0("t", seq_len(n))
  }
  melted <- melt(lat[n:1, ], na.rm = TRUE)
  pl <- ggplot(data = melted, aes(Var2, Var1, fill = value)) +
    geom_tile(color = "white") + ggtitle(title) +
    scale_fill_gradientn(colors = clrs, limit = c(-1, 1), space = "Lab") +
    theme_minimal() + coord_fixed() +
    theme(axis.title.x = element_blank(), axis.title.y = element_blank(),
          axis.text.x = element_text(color = col, size = 8, angle = 45, hjust = 1),
          axis.text.y = element_text(color = rev(col), size = 8),
          title = element_text(size = 10), panel.border = element_blank(),
          panel.background = element_blank(), axis.ticks = element_blank(),
          legend.justification = c(1, 0), legend.position = c(0.6, 0),
          legend.direction = "horizontal") +
    guides(fill = guide_colorbar(title = "", barwidth = 7, barheight = 1,
           title.position = "top", title.hjust = 0.5))
  if (remove_names) {
    pl <- pl + scale_x_discrete(labels = seq_len(n)) +
      scale_y_discrete(labels = rev(seq_len(n)))
  }
  pl
}

dat <- readRDS(argv$data)
if (nzchar(argv$name)) {
  if (is.null(dat[[argv$name]])) stop("Cannot find '", argv$name, "' in ", argv$data)
  dat <- dat[[argv$name]]
}
if (is.null(names(dat$U))) names(dat$U) <- paste0("Comp_", seq_along(dat$U))

# Align weights to the covariance components by name: get_estimated_pi() returns
# one extra entry for the null component, so `$w` is one longer than `$U`.
w <- if (!is.null(names(dat$w))) dat$w[names(dat$U)] else dat$w[seq_along(dat$U)]
w[is.na(w)] <- 0
meta <- data.frame(U = names(dat$U), w = as.numeric(w), stringsAsFactors = FALSE)
n_comp <- length(meta$U[which(meta$w > argv$tol)])
meta <- head(meta[order(meta$w, decreasing = TRUE), ],
             if (argv$max_comp > 1L) argv$max_comp else nrow(meta))
message(sprintf("%d of %d components have weight > %g", n_comp, length(dat$w), argv$tol))

res <- list()
for (i in seq_len(n_comp)) {
  title <- paste(meta$U[i], "w =", round(meta$w[i], 6))
  m <- dat$U[[meta$U[i]]]
  m <- if (is.list(m)) m$mat else m           # updated udr structure carries $mat
  if (is.matrix(m)) {
    res[[length(res) + 1L]] <- plot_sharing(m, to_cor = argv$to_cor, title = title,
                                            remove_names = argv$remove_label)
  }
}

unit <- 4; n_col <- 5; n_row <- ceiling(length(res) / n_col)
dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
pdf(argv$output, width = unit * n_col, height = unit * n_row)
do.call(gridExtra::grid.arrange,
        c(res, list(ncol = n_col, nrow = n_row,
                    bottom = sprintf("Data source: %s", basename(argv$data)))))
dev.off()
cat(sprintf("Wrote %d covariance heatmap(s) to %s\n", length(res), argv$output))
