#!/usr/bin/env Rscript
# fine_mapping_vcf.R
#
# Write a fine-mapping VCF (per-variant ES / CS / PIP in the sample column) from
# a QtlFineMappingResult / GwasFineMappingResult via pecotmr::writeSumstatsVcf.
# Replaces the legacy inline create_vcf() + VariantAnnotation::writeVcf in the
# mv_susie / uni_susie cells: the VCF assembly now lives in pecotmr's vcfWriter.
#
# Inputs:
#   --input             FineMappingResult RDS.
#   --output            Output VCF path (bgzipped + indexed when it ends .bgz/.gz).
#   --sample-name       VCF sample ("Studies") name. Default: the FMR study.
#   --split-by-context  Write one VCF per context (required when the collection
#                       has >1 row and no single row is selected).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Write a fine-mapping VCF from a FineMappingResult")
parser <- add_argument(parser, "--input", help = "FineMappingResult RDS", type = "character")
parser <- add_argument(parser, "--output", help = "Output VCF path (.bgz/.gz => bgzipped+indexed)",
                       type = "character")
parser <- add_argument(parser, "--sample-name", help = "VCF sample/Studies name",
                       type = "character", default = NA)
parser <- add_argument(parser, "--split-by-context", help = "Write one VCF per context",
                       flag = TRUE)
argv <- parse_args(parser)

fmr <- readRDS(argv$input)
if (!methods::is(fmr, "FineMappingResultBase"))
  stop("--input must be a FineMappingResult RDS (got '", class(fmr)[[1L]], "').")

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
out <- writeSumstatsVcf(
  fmr, outputPath = argv$output,
  sampleName = if (is.na(argv$sample_name)) NULL else argv$sample_name,
  splitByContext = argv$split_by_context)
cat(sprintf("Wrote %d fine-mapping VCF(s) from %s to %s\n",
            length(out), basename(argv$input), argv$output))
