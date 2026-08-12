#!/usr/bin/env Rscript
# ============================================================
# metal.R
# Mirrors: code/SoS/multivariate_genome/METAL/METAL.ipynb
#   (post-processing steps METAL_3 Output_reformatting and METAL_4 recipe)
#
# The METAL binary itself runs in the notebook; this worker does the R/Python-free
# post-processing:
#   --step reformat : raw METAL output -> reformatted .METAL.txt + a GWAS-VCF (.vcf.bgz)
#   --step recipe   : build the {name}.METAL_list.txt (#chr -> per-chrom sumstat file)
# ============================================================
suppressPackageStartupMessages({
  library(optparse)
  library(vroom)
  library(dplyr)
  library(tidyr)
})

opt_list <- list(
  make_option("--step",            type = "character", default = NULL, help = "reformat | recipe"),
  make_option("--input",           type = "character", default = NULL,
              help = "[reformat] raw METAL output (.1.METAL.txt)"),
  make_option("--output-sumstat",  type = "character", default = NULL,
              help = "[reformat] reformatted sumstat .METAL.txt"),
  make_option("--output-vcf",      type = "character", default = NULL,
              help = "[reformat] GWAS-VCF output (.METAL.vcf.bgz)"),
  make_option("--name",            type = "character", default = NULL,
              help = "study/meta name (VCF sample column + recipe value column)"),
  make_option("--sumstat-list",    type = "character", default = NULL,
              help = "[recipe] original sumstat_list tsv (supplies the #chr column)"),
  make_option("--output",          type = "character", default = NULL,
              help = "[recipe] output METAL_list.txt")
)
parsed <- parse_args(OptionParser(option_list = opt_list), positional_arguments = TRUE)
opt <- parsed$options
pos_args <- parsed$args
if (is.null(opt$step)) stop("--step is required (reformat | recipe)")

# Build a GWAS-VCF (ES/SE/LP per sample column). Ported verbatim from the notebook
# (modified from the gwasvcf package).
create_vcf <- function(chrom, pos, nea, ea, snp = NULL, ea_af = NULL, effect = NULL,
                       se = NULL, pval = NULL, n = NULL, ncase = NULL, name = NULL) {
  stopifnot(length(chrom) == length(pos))
  if (is.null(snp)) snp <- paste0(chrom, ":", pos)
  snp <- paste0(chrom, ":", pos)
  nsnp <- length(chrom)
  gen <- list()
  if (!is.null(ea_af)) gen[["AF"]] <- matrix(ea_af, nsnp)
  if (!is.null(effect)) gen[["ES"]] <- matrix(effect, nsnp)
  if (!is.null(se))     gen[["SE"]] <- matrix(se, nsnp)
  if (!is.null(pval))   gen[["LP"]] <- matrix(-log10(pval), nsnp)
  if (!is.null(n))      gen[["SS"]] <- matrix(n, nsnp)
  if (!is.null(ncase))  gen[["NC"]] <- matrix(ncase, nsnp)
  gen <- S4Vectors::SimpleList(gen)
  gr <- GenomicRanges::GRanges(chrom, IRanges::IRanges(
    start = pos, end = pos + pmax(nchar(nea), nchar(ea)) - 1, names = snp))
  coldata <- S4Vectors::DataFrame(Studies = name, row.names = name)
  hdr <- VariantAnnotation::VCFHeader(
    header = IRanges::DataFrameList(fileformat = S4Vectors::DataFrame(
      Value = "VCFv4.2", row.names = "fileformat")), sample = name)
  VariantAnnotation::geno(hdr) <- S4Vectors::DataFrame(
    Number = c("A", "A", "A", "A", "A", "A"),
    Type = c("Float", "Float", "Float", "Float", "Float", "Float"),
    Description = c("Effect size estimate relative to the alternative allele",
                   "Standard error of effect size estimate",
                   "-log10 p-value for effect estimate",
                   "Alternate allele frequency in the association study",
                   "Sample size used to estimate genetic effect",
                   "Number of cases used to estimate genetic effect"),
    row.names = c("ES", "SE", "LP", "AF", "SS", "NC"))
  VariantAnnotation::geno(hdr) <- subset(
    VariantAnnotation::geno(hdr),
    rownames(VariantAnnotation::geno(hdr)) %in% names(gen))
  vcf <- VariantAnnotation::VCF(rowRanges = gr, colData = coldata,
                                exptData = list(header = hdr), geno = gen)
  VariantAnnotation::alt(vcf) <- Biostrings::DNAStringSetList(as.list(ea))
  VariantAnnotation::ref(vcf) <- Biostrings::DNAStringSet(nea)
  VariantAnnotation::fixed(vcf)$FILTER <- "PASS"
  sort(vcf)
}

run_reformat <- function(opt) {
  for (f in c("input", "output-sumstat", "output-vcf", "name"))
    if (is.null(opt[[f]])) stop(sprintf("--%s is required for --step reformat", f))

  # METAL's output header is discarded; columns are taken positionally.
  raw <- vroom(opt$input, delim = "\t", skip = 1,
               col_names = c("variant_id", "alt", "ref", "beta", "se", "pval", "Direction"),
               col_types = cols(variant_id = col_character(), alt = col_character(),
                                ref = col_character(), beta = col_double(),
                                se = col_double(), pval = col_double(),
                                Direction = col_character()))

  # Reformatted sumstat: add pos / chrom parsed from variant_id (chr:pos_ref_alt).
  reformatted <- raw %>%
    mutate(pos   = sub("_.*", "", sub("^[^:]*:", "", variant_id)),
           chrom = sub(":.*", "", variant_id))
  vroom_write(reformatted, opt$`output-sumstat`, delim = "\t", na = "")

  # GWAS-VCF: alleles come from the variant_id (chr:pos_ref_alt), effect/se from METAL.
  ids <- tibble(snps = raw$variant_id) %>%
    separate(snps, into = c("chr", "pos_ref_alt"), sep = ":", remove = FALSE) %>%
    separate(pos_ref_alt, into = c("pos", "ref", "alt"), sep = "_") %>%
    mutate(chr = as.numeric(chr), pos = as.numeric(pos))
  vcf <- create_vcf(chrom = ids$chr, pos = ids$pos, ea = ids$alt, nea = ids$ref,
                    effect = raw$beta, se = raw$se, name = opt$name)
  VariantAnnotation::writeVcf(vcf, sub("\\.bgz$|\\.gz$", "", opt$`output-vcf`), index = TRUE)
}

run_recipe <- function(opt, sumstat_files) {
  for (f in c("sumstat-list", "name", "output"))
    if (is.null(opt[[f]])) stop(sprintf("--%s is required for --step recipe", f))
  if (length(sumstat_files) < 1)
    stop("recipe needs the per-chrom sumstat files as positional arguments")
  chr <- vroom(opt$`sumstat-list`, delim = "\t",
               col_types = cols(.default = col_character()))[["#chr"]]
  recipe <- tibble(`#chr` = chr)
  recipe[[opt$name]] <- sumstat_files
  vroom_write(recipe, opt$output, delim = "\t")
}

switch(opt$step,
  reformat = run_reformat(opt),
  recipe   = run_recipe(opt, pos_args),
  stop(sprintf("Unknown step '%s'. Available: reformat, recipe", opt$step))
)
