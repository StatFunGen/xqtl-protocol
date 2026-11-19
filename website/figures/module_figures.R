# Teaching figures for the module pages, drawn from simulated data in the xqtl-paper style
# (theme_bw, no grid lines, bold titles, soft colour ramps).
# Run from the repository root:  Rscript website/figures/module_figures.R
# Writes website/figures/modules/<module>.png
suppressPackageStartupMessages({ library(ggplot2); library(grid) })
set.seed(2026)
out <- "website/figures/modules"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

navy <- "#19354a"; blue <- "#3d6289"; sky <- "#9fb8d0"; pale <- "#dce6ef"
red  <- "#b5523b"; sand <- "#d9a35b"; grey <- "#7d8790"
ann  <- 3.9                                              # annotation text size (mm)

theme_x <- function(base = 13) theme_bw(base_size = base) + theme(
  panel.grid = element_blank(),
  plot.title = element_text(face = "bold", size = base),
  axis.title = element_text(face = "bold"),
  legend.position = "bottom", legend.title = element_text(face = "bold"),
  strip.background = element_rect(fill = "#f3f6f8"), strip.text = element_text(face = "bold"))

# lay panels side by side with bold a, b, c tags (no tag for a single panel)
save_panels <- function(file, plots, widths = rep(1, length(plots)), w = 10, h = 4.2) {
  png(file.path(out, file), width = w, height = h, units = "in", res = 200)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, length(plots), widths = unit(widths, "null"))))
  for (i in seq_along(plots)) {
    vp <- viewport(layout.pos.row = 1, layout.pos.col = i)
    print(plots[[i]], vp = vp)
    pushViewport(vp)
    if (length(plots) > 1) grid.text(letters[i], x = unit(4, "pt"), y = unit(1, "npc") - unit(4, "pt"),
              just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 16))
    popViewport()
  }
  invisible(dev.off())
}

# Genotypes with block-structured LD: inside a block each haplotype copies one of a few founder
# haplotypes, so LD is high within blocks, low between them, and not a simple function of distance.
# `link` makes one block's founder track the lead block's, so a distant variant can be in high LD.
sim_geno <- function(n, block_sizes, link = NULL, K = 4, err = 0.03) {
  hap <- function() {
    ch <- sapply(block_sizes, function(k) sample(K, n, replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1)))
    if (!is.null(link)) { same <- runif(n) < 0.9; ch[same, link[2]] <- ch[same, link[1]] }
    do.call(cbind, lapply(seq_along(block_sizes), function(b) {
      fnd <- founders[[b]]
      h <- fnd[ch[, b], , drop = FALSE]
      flip <- matrix(runif(length(h)) < err, nrow(h)); h[flip] <- 1L - h[flip]; h
    }))
  }
  founders <- lapply(block_sizes, function(k) {          # K founder haplotypes per block
    f <- matrix(rbinom(K * k, 1, 0.35), K); f[, colSums(f) %in% c(0, K)] <- c(1L, 0L, 0L, 0L); f })
  G <- hap() + hap()
  G[, apply(G, 2, var) > 0]
}

## 1. TensorQTL: one gene with a cis-eQTL --------------------------------------------------
n <- 300
bs <- c(18, 30, 12, 25, 40, 15, 22, 35, 28, 20, 33, 17, 26, 30, 24, 19, 26)
G <- sim_geno(n, bs, link = c(9, 3))                    # block 3 carries block 9's haplotype
pos <- sort(runif(ncol(G), -1e6, 1e6))
lead <- sum(bs[1:8]) + 6                                 # lead variant sits in block 9
expr <- 0.55 * G[, lead] + rnorm(n)
g <- G[, lead]
d1 <- data.frame(geno = factor(c("AA", "AG", "GG")[g + 1], levels = c("AA", "AG", "GG")), expr)
p1a <- ggplot(d1, aes(geno, expr)) +
  geom_jitter(width = 0.18, size = 0.9, colour = sky) +
  geom_boxplot(fill = NA, colour = navy, outlier.shape = NA, width = 0.5, linewidth = 0.6) +
  labs(title = "Expression by genotype", x = "Genotype at the lead variant",
       y = "Normalized expression") + theme_x()
lp <- apply(G, 2, function(x) -log10(summary(lm(expr ~ x))$coefficients[2, 4]))
r2 <- as.vector(cor(G, G[, lead]))^2
d1b <- data.frame(pos = pos / 1e6, lp, r2)[order(r2), ]
p1b <- ggplot(d1b, aes(pos, lp, colour = r2)) +
  geom_vline(xintercept = 0, linetype = 2, colour = grey) +
  geom_point(size = 1.6) +
  annotate("text", x = 0, y = max(lp) * 1.04, label = "gene start ", colour = grey, size = ann, hjust = 1) +
  scale_colour_gradient(low = pale, high = red, name = expression(bold(r^2)~"with lead"), limits = c(0, 1)) +
  labs(title = "Testing variants near a gene", x = "Position relative to gene start (Mb)",
       y = expression(bold(-log[10]~p))) + theme_x() +
  theme(legend.key.width = unit(1.2, "cm"))
save_panels("tensorqtl.png", list(p1a, p1b), widths = c(0.8, 1.2))

## 2. Hidden factors: choosing how many to keep -------------------------------------------
set.seed(2027)
k <- 1:30; ve <- 22 * exp(-k / 3.2) + 0.6 + runif(30, 0, 0.2)
p3a <- ggplot(data.frame(k, ve), aes(k, ve)) +
  geom_col(fill = sky, width = 0.75) + geom_col(data = data.frame(k = 1:8, ve = ve[1:8]), fill = blue, width = 0.75) +
  annotate("text", x = 9, y = ve[2], label = "first factors often capture\nbatch, cell mix, ...", hjust = 0, size = ann, colour = navy) +
  labs(title = "Variance explained per factor", x = "Hidden factor",
       y = "Variance explained (%)") + theme_x()
kk <- seq(0, 60, 5)
eg <- 1400 + 1600 * (1 - exp(-kk / 9)) - 14 * pmax(kk - 20, 0) + rnorm(length(kk), 0, 20)
best <- kk[which.max(eg)]
p3b <- ggplot(data.frame(kk, eg), aes(kk, eg)) +
  geom_vline(xintercept = best, linetype = 2, colour = red) +
  geom_line(colour = blue, linewidth = 0.8) + geom_point(colour = navy, size = 2) +
  annotate("text", x = best, y = min(eg), label = " most QTL genes in this simulation", hjust = 0, vjust = 0, colour = red, size = ann) +
  annotate("text", x = 60, y = min(eg) + 150, label = "too many factors can\nremove real signal", hjust = 1, vjust = 0, colour = grey, size = ann) +
  labs(title = "QTL genes found depend on the number", x = "Hidden factors added",
       y = "Genes with a QTL") + theme_x()
save_panels("hidden_factors.png", list(p3a, p3b))

## 3. Expression normalization: one gene, three scales -------------------------------------
set.seed(2028)
# Counts for 2,000 genes x 400 samples. In some samples a few genes take a large share of the
# reads (a composition effect), which plain library-size scaling gets wrong and TMM corrects.
ng <- 2000; ns <- 400
base <- rlnorm(ng, 3, 1.5)
lib <- rlnorm(ns, log(2e7), 0.35)
comp <- matrix(1, ng, ns); hot <- sample(ns, 120); comp[1:40, hot] <- 25
mu <- sweep(base * comp, 2, colSums(base * comp), "/") * rep(lib, each = ng)
Y <- matrix(rnbinom(ng * ns, mu = mu, size = 10), ng)
# TMM factors (Robinson & Oshlack 2010), written out so the script needs only ggplot2. On this
# matrix they equal edgeR::calcNormFactors(DGEList(Y), method = "TMM") to machine precision (edgeR 4.8.2).
tmm <- function(Y, logratio_trim = 0.3, sum_trim = 0.05) {
  N <- colSums(Y); uq <- apply(sweep(Y, 2, N, "/"), 2, quantile, 0.75)
  ref <- which.min(abs(uq - mean(uq)))
  f <- sapply(seq_len(ncol(Y)), function(j) {
    y <- Y[, j]; r <- Y[, ref]; ok <- y > 0 & r > 0
    M <- log2((y[ok] / N[j]) / (r[ok] / N[ref])); A <- 0.5 * log2((y[ok] / N[j]) * (r[ok] / N[ref]))
    w <- (N[j] - y[ok]) / (N[j] * y[ok]) + (N[ref] - r[ok]) / (N[ref] * r[ok])
    keep <- rank(M) > length(M) * logratio_trim & rank(M) <= length(M) * (1 - logratio_trim) &
            rank(A) > length(A) * sum_trim & rank(A) <= length(A) * (1 - sum_trim)
    2^(sum(M[keep] / w[keep]) / sum(1 / w[keep]))
  })
  f / exp(mean(log(f)))
}
eff <- colSums(Y) * tmm(Y)
gi <- 500                                                 # an ordinary gene, not one of the 40 "hot" ones
cnt <- Y[gi, ]
cpm <- log2((cnt + 0.5) / (eff + 1) * 1e6)
inv <- qnorm((rank(cpm) - 0.5) / length(cpm))
d4 <- rbind(data.frame(stage = "1. Raw read counts", v = cnt),
            data.frame(stage = "2. TMM + log2 CPM", v = cpm),
            data.frame(stage = "3. Inverse-normal", v = inv))
d4$stage <- factor(d4$stage, levels = unique(d4$stage))
p4 <- ggplot(d4, aes(v)) +
  geom_histogram(bins = 35, fill = sky, colour = "white", linewidth = 0.2) +
  facet_wrap(~stage, scales = "free") +
  labs(title = "One gene across 400 samples, after each step", x = NULL, y = "Samples") +
  theme_x()
save_panels("expression_normalization.png", list(p4), h = 3.8)
