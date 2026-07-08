#!/usr/bin/env Rscript
# sldsc_postprocess.R
#
# Post-process polyfun's per-trait sLDSC outputs via
# pecotmr::sldscPostprocessingPipeline(). All file I/O (annotation table,
# reference-panel allele frequencies, per-trait single/joint polyfun runs)
# happens here, at the I/O boundary, through the pecotmr reader functions
# (readSldscAnnot / readSldscFrq / readSldscTrait); the pipeline itself is
# pure computation over the in-memory SldscData object it returns.
#
# For each trait, the wrapper auto-detects the single-target run
# directories (`<annotation_name>_single_<i>/`) and the optional
# joint-target directory (`<annotation_name>_joint/`) under
# --heritability-cwd, reads the polyfun triples (.results / .log /
# .part_delete) at `<dir>/<trait>`, assembles the per-trait
# list(single = list(<runs>), joint = <run or NULL>) structure that
# SldscData() consumes, and runs the pipeline.
#
# Inputs:
#   --traits-file <txt>          One trait sumstats filename per line.
#   --heritability-cwd <dir>     Parent directory of the [get_heritability]
#                                outputs (contains <annotation_name>_single_<i>/
#                                subdirs and optionally <annotation_name>_joint/).
#   --annotation-name <str>      Prefix used to detect the single/joint subdirs.
#   --target-anno-dir <dir>      Directory of target .annot.gz files (sd_C /
#                                binary detection; typically the joint dir,
#                                since it carries all target columns).
#   --frqfile-dir <dir>          Directory of reference-panel .frq files.
#                                Required when --maf-cutoff > 0.
#   --plink-name <str>           .frq filename prefix. Default "ADSP_chr".
#   --maf-cutoff <num>           MAF cutoff forwarded to the pipeline.
#                                Default 0.05; set 0 to opt out (no frq needed).
#   --target-categories c1,c2    Optional target annotation names. Auto-detected
#                                from the joint run when omitted.
#   --target-categories-label l1,l2
#                                Optional display names, same order as
#                                --target-categories; every "target" column /
#                                tau*-block colname is renamed to these.
#   --output <RDS>               Post-processing result RDS (per-trait tables +
#                                meta tables + params record).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Post-process polyfun sLDSC outputs via pecotmr::sldscPostprocessingPipeline")
parser <- add_argument(parser, "--traits-file",
                       help = "Text file: one trait sumstats filename per line",
                       type = "character")
parser <- add_argument(parser, "--heritability-cwd",
                       help = "Parent dir of [get_heritability] outputs",
                       type = "character")
parser <- add_argument(parser, "--annotation-name",
                       help = "Prefix used to detect single/joint subdirs",
                       type = "character")
parser <- add_argument(parser, "--target-anno-dir",
                       help = "Directory of target .annot.gz files",
                       type = "character")
parser <- add_argument(parser, "--frqfile-dir",
                       help = "Directory of reference-panel .frq files",
                       type = "character", default = "")
parser <- add_argument(parser, "--plink-name",
                       help = ".frq filename prefix", type = "character",
                       default = "ADSP_chr")
parser <- add_argument(parser, "--maf-cutoff",
                       help = "MAF cutoff forwarded to the pipeline",
                       type = "numeric", default = 0.05)
parser <- add_argument(parser, "--target-categories",
                       help = "Comma-separated target annotation names",
                       type = "character", default = "")
parser <- add_argument(parser, "--target-categories-label",
                       help = "Comma-separated display names for target categories",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output post-processing RDS path",
                       type = "character")
argv <- parse_args(parser)

splitCsv <- function(x) {
  if (!nzchar(x)) return(character(0))
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

traits <- readLines(argv$traits_file)
traits <- traits[nzchar(trimws(traits))]
if (length(traits) == 0L)
  stop("--traits-file is empty: ", argv$traits_file)

targetCats <- splitCsv(argv$target_categories)
targetLab  <- splitCsv(argv$target_categories_label)

# Auto-detect single-target and joint-target output directories.
herRoot    <- argv$heritability_cwd
allSubdirs <- list.dirs(herRoot, recursive = FALSE)
singlePattern <- paste0("^", argv$annotation_name, "_single_([0-9]+)$")
jointName     <- paste0(argv$annotation_name, "_joint")
singleDirs    <- allSubdirs[grepl(singlePattern, basename(allSubdirs))]
singleIdx     <- as.integer(sub(singlePattern, "\\1", basename(singleDirs)))
singleDirs    <- singleDirs[order(singleIdx)]
jointDir      <- file.path(herRoot, jointName)
hasJoint      <- dir.exists(jointDir)

if (length(singleDirs) == 0L)
  stop("No single-target dirs matching '", singlePattern, "' under ", herRoot)

message(sprintf("Detected %d single-target dir(s)%s",
                length(singleDirs),
                if (hasJoint) "; joint-target dir present" else "; no joint-target dir"))

# Read the polyfun triples (.results/.log/.part_delete) at <dir>/<trait> and
# assemble the per-trait single/joint run structure SldscData() consumes.
traitList <- setNames(lapply(traits, function(t) {
  list(
    single = lapply(singleDirs, function(d) readSldscTrait(file.path(d, t))),
    joint  = if (hasJoint) readSldscTrait(file.path(jointDir, t)) else NULL
  )
}), traits)

# Read the annotation table and (when MAF filtering is on) the reference frq.
annot <- readSldscAnnot(argv$target_anno_dir)
frq <- NULL
if (argv$maf_cutoff > 0) {
  if (!nzchar(argv$frqfile_dir))
    stop("--maf-cutoff = ", argv$maf_cutoff,
         " requires --frqfile-dir (frq data needed for MAF filtering).")
  frq <- readSldscFrq(argv$frqfile_dir, plinkName = argv$plink_name)
} else if (nzchar(argv$frqfile_dir)) {
  frq <- readSldscFrq(argv$frqfile_dir, plinkName = argv$plink_name)
}

sldscData <- SldscData(annot = annot, frq = frq, traits = traitList)

res <- sldscPostprocessingPipeline(
  sldscData,
  mafCutoff        = argv$maf_cutoff,
  targetCategories = if (length(targetCats) > 0) targetCats else NULL,
  targetLabels     = if (length(targetLab)  > 0) targetLab  else NULL)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
message("S-LDSC post-processing complete; results written to ", argv$output)
