#!/usr/bin/env Rscript
# ============================================================
# qtl_sumstats_construct.R
# Standalone CLI worker for qtl_rss_analysis.ipynb [generate_qtl_sumstats].
#
# Builds one `pecotmr::QtlSumStats` per (study, context, trait) from a cis-QTL
# nominal association table, attaches an LD reference panel as the ldSketch and
# runs `summaryStatsQc()`. The result is the summary-statistics counterpart of a
# QtlDataset: everything downstream (RSS fine-mapping, RSS TWAS weights) reads
# it instead of individual-level genotypes.
#
# This is the QTL sibling of gwas_sumstats_construct.R and is just as thin:
# pecotmr's loadQtlSumStatsFromManifest() does the reading, column resolution,
# trait/region restriction and construction; summaryStatsQc() does the QC.
#
# Two details of a cis-QTL scan that the loader is told about explicitly:
#   --trait-column       the scan writes EVERY gene into one table, so the rows
#                        for this trait are selected by that column. (Region
#                        restriction cannot do it: neighbouring cis windows
#                        overlap, and it needs a tabix index besides.)
#   --variant-id-alleles the scan reports effect + se against the counted
#                        allele and leaves the alleles inside the variant id.
#                        The order within an id cannot be recovered from the
#                        string -- a .pvar writes REF:ALT (pecotmr's canonical
#                        A2:A1) while a PLINK .bim writes A1:A2 -- so declare
#                        which one this file uses. Declaring it wrong is
#                        SILENT: summaryStatsQc() "corrects" the apparent
#                        mismatch by sign- and strand-flipping every variant
#                        against the panel. A real A1/A2 column always wins.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Build a QtlSumStats (+ LD sketch, + QC) from a cis-QTL nominal table")
parser <- add_argument(parser, "--sumstats", type = "character", default = "",
                       help = "cis-QTL nominal association table (TensorQTL *.cis_qtl.pairs.tsv.gz)")
parser <- add_argument(parser, "--study", type = "character", default = "",
                       help = "Study label")
parser <- add_argument(parser, "--context", type = "character", default = "",
                       help = "Context (condition/tissue) label")
parser <- add_argument(parser, "--trait", type = "character", default = "",
                       help = "Trait (gene) id")
parser <- add_argument(parser, "--trait-column", type = "character", default = "molecular_trait_id",
                       help = "Column naming each row's trait; empty disables the filter (single-trait file)")
parser <- add_argument(parser, "--variant-id-alleles", type = "character", default = "none",
                       help = "Where the alleles live when there is no A1/A2 column: none | A2A1 | A1A2")
parser <- add_argument(parser, "--ld-sketch", type = "character", default = "",
                       help = "LD reference panel: a genotype path/prefix, or a per-chromosome LD meta file")
parser <- add_argument(parser, "--genome", type = "character", default = "GRCh38",
                       help = "Genome build recorded on the collection")
parser <- add_argument(parser, "--region", type = "character", default = "",
                       help = "Optional chr:start-end restriction (needs a tabix-indexed sumstats file)")
parser <- add_argument(parser, "--n-sample", type = "numeric", default = NA,
                       help = "Study-level total N, used when the file has no per-variant N column")
parser <- add_argument(parser, "--column-mapping", type = "character", default = "",
                       help = "Optional column-mapping YAML for non-standard source column names")
parser <- add_argument(parser, "--maf", type = "numeric", default = 0,
                       help = "summaryStatsQc mafCutoff (study and LD panel)")
parser <- add_argument(parser, "--mac", type = "numeric", default = 0,
                       help = "summaryStatsQc macCutoff")
parser <- add_argument(parser, "--imiss", type = "numeric", default = 1,
                       help = "summaryStatsQc imissCutoff")
parser <- add_argument(parser, "--z-mismatch-qc", type = "character", default = "none",
                       help = "summaryStatsQc zMismatchQc: none | slalom | dentist")
parser <- add_argument(parser, "--pip-cutoff-to-skip", type = "numeric", default = 0,
                       help = "summaryStatsQc pipCutoffToSkip; 0 = off")
parser <- add_argument(parser, "--skip-qc", flag = TRUE,
                       help = "Serialise the raw QtlSumStats without summaryStatsQc() (diagnostics only)")
parser <- add_argument(parser, "--output", type = "character", default = "",
                       help = "Output RDS")
argv <- parse_args(parser)

for (a in c("sumstats", "study", "context", "trait", "ld_sketch", "output"))
  if (!nzchar(argv[[a]]))
    stop("--", gsub("_", "-", a), " is required")

# One manifest row per (study, context, trait); the loader takes it from here.
manifest <- data.frame(
  study        = argv$study,
  context      = argv$context,
  trait        = argv$trait,
  sumStatsPath = argv$sumstats,
  stringsAsFactors = FALSE)
if (nzchar(argv$column_mapping)) manifest$columnMapping <- argv$column_mapping
if (!is.na(argv$n_sample))       manifest$nSample      <- argv$n_sample

qss <- loadQtlSumStatsFromManifest(
  manifest         = manifest,
  genome           = argv$genome,
  ldSketch         = argv$ld_sketch,
  region           = if (nzchar(argv$region)) argv$region else NULL,
  columnMapping    = if (nzchar(argv$column_mapping)) argv$column_mapping else NULL,
  traitColumn      = if (nzchar(argv$trait_column)) argv$trait_column else NULL,
  variantIdAlleles = argv$variant_id_alleles)

out <- if (argv$skip_qc) {
  message("--skip-qc set; serialising the raw QtlSumStats without summaryStatsQc().")
  qss
} else {
  summaryStatsQc(qss,
                 mafCutoff       = argv$maf,
                 macCutoff       = argv$mac,
                 imissCutoff     = argv$imiss,
                 zMismatchQc     = argv$z_mismatch_qc,
                 pipCutoffToSkip = argv$pip_cutoff_to_skip)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(out, argv$output, compress = "xz")
cat(sprintf("Wrote QtlSumStats for %s/%s/%s (%d variant(s), LD sketch %s) to %s\n",
            argv$study, argv$context, argv$trait, length(out[[1L]]),
            if (is.null(getLdSketch(out))) "NULL" else
              paste(dim(getLdSketch(out)), collapse = " x "),
            argv$output))
