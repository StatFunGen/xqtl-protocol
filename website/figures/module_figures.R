# Teaching figures for the module pages, drawn from simulated data in the xqtl-paper style
# (theme_bw, no grid lines, bold titles, soft colour ramps).
# Run from the repository root:  Rscript website/figures/module_figures.R
# Writes website/figures/modules/<module>.png
suppressPackageStartupMessages({ library(ggplot2); library(grid) })
set.seed(2026)
out <- "website/figures/modules"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

navy <- "#19354a"; blue <- "#3d6289"; sky <- "#9fb8d0"; pale <- "#dce6ef"
red  <- "#b5523b"; sand <- "#d9a35b"; grey <- "#9aa3ab"

theme_x <- function(base = 12) theme_bw(base_size = base) + theme(
  panel.grid = element_blank(),
  plot.title = element_text(face = "bold", size = base),
  axis.title = element_text(face = "bold"),
  legend.position = "bottom", legend.title = element_text(face = "bold"),
  strip.background = element_rect(fill = "#f3f6f8"), strip.text = element_text(face = "bold"))

# lay panels side by side with bold a, b, c tags
save_panels <- function(file, plots, widths = rep(1, length(plots)), w = 10, h = 4) {
  png(file.path(out, file), width = w, height = h, units = "in", res = 200)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, length(plots), widths = unit(widths, "null"))))
  for (i in seq_along(plots)) {
    vp <- viewport(layout.pos.row = 1, layout.pos.col = i)
    print(plots[[i]], vp = vp)
    pushViewport(vp)
    if (length(plots) > 1) grid.text(letters[i], x = unit(4, "pt"), y = unit(1, "npc") - unit(4, "pt"),
              just = c("left", "top"), gp = gpar(fontface = "bold", fontsize = 15))
    popViewport()
  }
  invisible(dev.off())
}

## 1. TensorQTL: one gene with a cis-eQTL --------------------------------------------------
n <- 300; maf <- 0.3
g <- rbinom(n, 2, maf)
expr <- 0.6 * g + rnorm(n)
d1 <- data.frame(geno = factor(c("AA", "AG", "GG")[g + 1], levels = c("AA", "AG", "GG")), expr)
p1a <- ggplot(d1, aes(geno, expr)) +
  geom_jitter(width = 0.18, size = 0.9, colour = sky) +
  geom_boxplot(fill = NA, colour = navy, outlier.shape = NA, width = 0.5, linewidth = 0.6) +
  labs(title = "Expression rises with each G allele", x = "Genotype at the lead variant",
       y = "Normalized expression") + theme_x()

pos <- sort(runif(400, -1e6, 1e6)); lead <- which.min(abs(pos - 1.2e5))
ld <- exp(-abs(pos - pos[lead]) / 6e4)                # LD with the lead variant decays with distance
z  <- 9 * ld * sign(rnorm(400, 2)) + rnorm(400)
d1b <- data.frame(pos = pos / 1e6, lp = -log10(2 * pnorm(-abs(z))), ld = ld^2)
p1b <- ggplot(d1b, aes(pos, lp, colour = ld)) +
  annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#f3f6f8") +
  geom_vline(xintercept = 0, linetype = 2, colour = grey) +
  geom_point(size = 1.4) +
  annotate("text", x = 0, y = max(d1b$lp) * 1.04, label = "gene start ", colour = grey, size = 3.4, hjust = 1) +
  scale_colour_gradient(low = pale, high = red, name = expression(bold(r^2)~"with lead")) +
  labs(title = "Every variant within 1 Mb is tested", x = "Position relative to gene start (Mb)",
       y = expression(bold(-log[10]~p))) + theme_x() +
  theme(legend.key.width = unit(1.2, "cm"))
save_panels("tensorqtl.png", list(p1a, p1b), widths = c(0.8, 1.2))

## 2. QTL post-processing: local then global correction -----------------------------------
m <- 2000; is_e <- runif(m) < 0.25
pmin <- ifelse(is_e, rbeta(m, 0.08, 1), runif(m))       # gene-level adjusted minimum p
o <- order(pmin); bh <- p.adjust(pmin, "BH")
d2 <- data.frame(p = pmin, sig = bh < 0.05)
p2a <- ggplot(d2, aes(p, fill = sig)) +
  geom_histogram(breaks = seq(0, 1, 0.025), colour = "white", linewidth = 0.2) +
  geom_hline(yintercept = sum(!is_e) / 40, linetype = 2, colour = grey) +
  annotate("text", x = 0.62, y = sum(!is_e) / 40, label = "genes with no QTL\n(flat background)",
           vjust = -1.6, colour = grey, size = 3.4) +
  scale_fill_manual(values = c(`TRUE` = red, `FALSE` = sky), labels = c(`TRUE` = "FDR < 5%", `FALSE` = "not significant"), name = NULL) +
  labs(title = "Step 2: one adjusted p-value per gene", x = "Gene-level p-value (after local adjustment)",
       y = "Genes") + theme_x()
nv <- 60; zv <- 3.2 * exp(-abs(1:nv - 25) / 6) + rnorm(nv, 0, 0.5)
pv <- 2 * pnorm(-abs(zv)); keep <- pv < 0.05 / 8        # eigenMT-like: ~8 effective tests
d2b <- data.frame(i = 1:nv, lp = -log10(pv), keep)
p2b <- ggplot(d2b, aes(i, lp, colour = keep)) +
  geom_hline(yintercept = -log10(0.05 / 8), linetype = 2, colour = red) +
  annotate("text", x = nv, y = -log10(0.05 / 8), label = "local threshold", hjust = 1, vjust = -0.5, colour = red, size = 3.4) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = c(`TRUE` = red, `FALSE` = sky), labels = c(`TRUE` = "reported xQTL", `FALSE` = "not reported"), name = NULL) +
  labs(title = "Steps 1 and 3: variants within a significant gene", x = "Variants in the cis window (ordered by position)",
       y = expression(bold(-log[10]~p))) + theme_x()
save_panels("qtl_postprocessing.png", list(p2b, p2a))

## 3. Hidden factors: how many to keep -----------------------------------------------------
k <- 1:30; ve <- 22 * exp(-k / 3.2) + 0.6 + runif(30, 0, 0.2)
p3a <- ggplot(data.frame(k, ve), aes(k, ve)) +
  geom_col(fill = sky, width = 0.75) + geom_col(data = data.frame(k = 1:8, ve = ve[1:8]), fill = blue, width = 0.75) +
  annotate("text", x = 9, y = ve[2], label = "first factors capture\nbatch, cell mix, ...", hjust = 0, size = 3.4, colour = navy) +
  labs(title = "Hidden structure is concentrated in a few factors", x = "Hidden factor",
       y = "Variance explained (%)") + theme_x()
kk <- seq(0, 60, 5); eg <- 1400 + 1700 * (1 - exp(-kk / 12)) - 6 * pmax(kk - 25, 0) + rnorm(length(kk), 0, 25)
p3b <- ggplot(data.frame(kk, eg), aes(kk, eg)) +
  geom_line(colour = blue, linewidth = 0.8) + geom_point(colour = navy, size = 2) +
  annotate("text", x = 0, y = eg[1], label = "known covariates only", hjust = -0.12, vjust = 0.5, size = 3.4, colour = grey) +
  labs(title = "Adding factors finds more QTL genes, then levels off", x = "Hidden factors added",
       y = "Genes with a QTL") + theme_x()
save_panels("hidden_factors.png", list(p3a, p3b))

## 4. Expression normalization: one gene, three scales -------------------------------------
lib <- rlnorm(400, log(2e7), 0.35); mu <- 40 * lib / 2e7
cnt <- rnbinom(400, mu = mu, size = 4)
cpm <- log2((cnt + 0.5) / lib * 1e6)
inv <- qnorm((rank(cpm) - 0.5) / length(cpm))
d4 <- rbind(data.frame(stage = "1. Raw read counts", v = cnt),
            data.frame(stage = "2. TMM + log CPM", v = cpm),
            data.frame(stage = "3. Inverse-normal", v = inv))
d4$stage <- factor(d4$stage, levels = unique(d4$stage))
p4 <- ggplot(d4, aes(v)) +
  geom_histogram(bins = 35, fill = sky, colour = "white", linewidth = 0.2) +
  facet_wrap(~stage, scales = "free") +
  labs(title = "The same gene across 400 samples, before and after each step", x = NULL, y = "Samples") +
  theme_x()
save_panels("expression_normalization.png", list(p4), h = 3.6)

## 5. Methylation: beta and M values -------------------------------------------------------
beta <- c(rbeta(6000, 1.2, 14), rbeta(7000, 14, 1.6), rbeta(1500, 4, 4))
d5 <- data.frame(beta)
p5a <- ggplot(d5, aes(beta)) + geom_density(fill = pale, colour = blue, linewidth = 0.7) +
  annotate("text", x = 0.05, y = 2.6, label = "unmethylated", hjust = 0, size = 3.4, colour = navy) +
  annotate("text", x = 0.97, y = 2.6, label = "methylated", hjust = 1, size = 3.4, colour = navy) +
  labs(title = "Most CpG sites are fully on or off", x = "Beta value (methylated / total signal)",
       y = "Density") + theme_x()
bb <- seq(0.005, 0.995, length.out = 200)
p5b <- ggplot(data.frame(bb, M = log2(bb / (1 - bb))), aes(bb, M)) +
  annotate("rect", xmin = 0, xmax = 0.1, ymin = -Inf, ymax = Inf, fill = "#f3f6f8") +
  annotate("rect", xmin = 0.9, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#f3f6f8") +
  geom_line(colour = red, linewidth = 0.9) +
  annotate("text", x = 0.5, y = 5, label = "M stretches the crowded ends,\nso tests behave like ordinary regression",
           size = 3.4, colour = navy) +
  labs(title = "M value = log2(beta / (1 - beta))", x = "Beta value", y = "M value") + theme_x()
save_panels("methylation.png", list(p5a, p5b))

## 6. RNA-seq calling: where the reads go --------------------------------------------------
ns <- 12; s <- sprintf("S%02d", 1:ns)
uniq <- runif(ns, 0.78, 0.9); multi <- runif(ns, 0.04, 0.09); unm <- 1 - uniq - multi
uniq[9] <- 0.52; multi[9] <- 0.08; unm[9] <- 0.40                     # one poor sample
d6 <- data.frame(sample = rep(s, 3), frac = c(uniq, multi, unm),
                 fate = factor(rep(c("Uniquely mapped", "Multi-mapped", "Unmapped"), each = ns),
                               levels = c("Unmapped", "Multi-mapped", "Uniquely mapped")))
p6a <- ggplot(d6, aes(sample, frac, fill = fate)) + geom_col(width = 0.8) +
  annotate("text", x = 9, y = 1.04, label = "check this sample", size = 3.4, colour = red) +
  scale_fill_manual(values = c(`Uniquely mapped` = blue, `Multi-mapped` = sky, Unmapped = sand), name = NULL) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
  labs(title = "STAR alignment: share of reads per sample", x = "Sample", y = "Reads") + theme_x()
tpm <- c(rlnorm(9000, 0.5, 2.2)); tpm[runif(9000) < 0.3] <- 0
d6b <- data.frame(l = log10(tpm + 0.01))
p6b <- ggplot(d6b, aes(l)) + geom_histogram(bins = 40, fill = sky, colour = "white", linewidth = 0.2) +
  geom_vline(xintercept = log10(0.1), linetype = 2, colour = red) +
  annotate("text", x = -0.85, y = 2200, label = "tall bar on the left:\ngenes not expressed (TPM 0)", hjust = 0, vjust = 1, size = 3.4, colour = grey, lineheight = 0.9) +
  annotate("text", x = log10(0.1), y = Inf, label = "TPM 0.1", hjust = -0.1, vjust = 1.5, size = 3.4, colour = red) +
  labs(title = "RNA-SeQC: genes per expression level", x = expression(bold(log[10]~"(TPM + 0.01)")),
       y = "Genes") + theme_x()
save_panels("rna_calling.png", list(p6a, p6b), widths = c(1.2, 0.9))
