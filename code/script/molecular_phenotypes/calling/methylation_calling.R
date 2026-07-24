#!/usr/bin/env Rscript
# ============================================================
# methylation_calling.R
# Standalone CLI worker for methylation_calling.ipynb steps.
#
# Steps (selected via --step):
#   sesame    — SeSAMe beta/M calling from IDATs ([sesame_1])
#   minfi     — minfi beta/M calling from IDATs ([minfi_1])
#   annotate  — probe->gene BED + bgzip/tabix ([*_2])
#
# Conventions: read/write with vroom, manipulate with dplyr, bgzip/tabix via
# Rsamtools (no data.table). Heavy Bioc packages load inside each step so
# --help works without them. CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
})

parser <- arg_parser("methylation_calling worker (see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "sesame | minfi | annotate")
parser <- add_argument(parser, "--sample-sheet", type = "character", default = "",
                       help = "[sesame/minfi] sample sheet CSV")
parser <- add_argument(parser, "--idat-folder", type = "character", default = "",
                       help = "[sesame] folder of IDATs (default: sample-sheet dir)")
parser <- add_argument(parser, "--sample-sheet-header-rows", type = "numeric", default = 0,
                       help = "[sesame] header rows to skip in the sample sheet")
parser <- add_argument(parser, "--samples-frac-dt-cutoff", type = "numeric", default = 0.8,
                       help = "[sesame] min probe detection-success fraction per sample")
parser <- add_argument(parser, "--n-cores", type = "numeric", default = 1,
                       help = "[sesame] BiocParallel workers (0 = auto)")
parser <- add_argument(parser, "--keep-only-cpg-probes", flag = TRUE,
                       help = "keep CpG-only probes (drop SNP probes)")
parser <- add_argument(parser, "--samples-pval-cutoff", type = "numeric", default = 0.05,
                       help = "[minfi] max mean detection p-value per sample")
parser <- add_argument(parser, "--probe-pval-cutoff", type = "numeric", default = 0.01,
                       help = "[minfi] max detection p-value per probe")
parser <- add_argument(parser, "--cross-reactive-probes", type = "character", default = "",
                       help = "[minfi] cross-reactive probe list (column 'probe')")
parser <- add_argument(parser, "--hg-build", type = "numeric", default = 38, help = "[minfi] 38 or 19")
parser <- add_argument(parser, "--output-rds", type = "character", default = "", help = "output RDS")
parser <- add_argument(parser, "--output-beta", type = "character", default = "", help = "output beta TSV")
parser <- add_argument(parser, "--output-m", type = "character", default = "", help = "output M TSV")
parser <- add_argument(parser, "--output-qcs", type = "character", default = "", help = "[sesame] output QC TSV")
parser <- add_argument(parser, "--input-beta", type = "character", default = "",
                       help = "[annotate] beta TSV")
parser <- add_argument(parser, "--input-m", type = "character", default = "", help = "[annotate] M TSV")
parser <- add_argument(parser, "--output-beta-bed", type = "character", default = "", help = "[annotate] beta BED.gz")
parser <- add_argument(parser, "--output-m-bed", type = "character", default = "", help = "[annotate] M BED.gz")
parser <- add_argument(parser, "--output-annot", type = "character", default = "", help = "[annotate] probe annotation TSV")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

mat_to_tsv <- function(m, path) {                        # matrix (rownames=ID) -> TSV
  df <- data.frame(ID = rownames(m), m, check.names = FALSE, row.names = NULL)
  vroom::vroom_write(df, path, delim = "\t")
}

# ---------------------------------------------------------------------------
sesame_step <- function(argv) {
  suppressPackageStartupMessages({ library(sesame); library(BiocParallel) })
  n_cores <- if (argv$n_cores == 0) BiocParallel::multicoreWorkers() else argv$n_cores
  bpp <- BiocParallel::MulticoreParam(n_cores)
  sesameData::sesameDataCache()
  proc <- if (argv$keep_only_cpg_probes) "QCDGPB" else "QCDPB"
  B2M <- function(x) { x[x == 0] <- min(x[x != 0]); x[x == 1] <- max(x[x != 1]); log2(x) - log2(1 - x) }

  idat_folder <- if (nzchar(argv$idat_folder)) argv$idat_folder else dirname(argv$sample_sheet)
  ss <- vroom::vroom(argv$sample_sheet, delim = ",", skip = argv$sample_sheet_header_rows,
                     show_col_types = FALSE)
  ss$well_name <- if ("Sentrix_Row_Column" %in% names(ss))
    paste(ss$Sentrix_ID, ss$Sentrix_Row_Column, sep = "_") else
    paste(ss$Sentrix_ID, ss$Sentrix_Position, sep = "_")
  ss$Sample_Name <- as.character(ss[[1]])

  sdfs <- openSesame(idat_folder, prep = "", func = NULL, BPPARAM = bpp)
  sdfs <- sdfs[names(sdfs) %in% ss$well_name]
  qcs <- openSesame(sdfs, prep = "", func = sesameQC_calcStats, BPPARAM = bpp)
  qcs_dt <- do.call(rbind, lapply(qcs, as.data.frame))
  qcs_dt <- data.frame(id = rownames(qcs_dt), qcs_dt, row.names = NULL)
  poor <- qcs_dt$id[qcs_dt$frac_dt < argv$samples_frac_dt_cutoff]
  sdfs <- sdfs[!(names(sdfs) %in% poor)]
  message(if (length(poor)) paste(poor, "removed (low quality)") else "No sample removed")

  beta <- openSesame(sdfs, prep = proc, BPPARAM = bpp)
  beta <- beta[rowSums(is.na(beta)) != ncol(beta), ]
  colnames(beta) <- ss$Sample_Name[match(colnames(beta), ss$well_name)]
  M <- B2M(beta)

  mat_to_tsv(beta, argv$output_beta)
  mat_to_tsv(M, argv$output_m)
  vroom::vroom_write(qcs_dt, argv$output_qcs, delim = "\t")
  saveRDS(list(sdfs = sdfs, qcs = qcs), argv$output_rds)
  message("sesame analysis completed!")
}

# ---------------------------------------------------------------------------
minfi_step <- function(argv) {
  suppressPackageStartupMessages({ library(minfi); library(stringr); library(tibble); library(dplyr) })
  cross_reactive <- vroom::vroom(argv$cross_reactive_probes, delim = "\t", show_col_types = FALSE)$probe
  targets <- read.metharray.sheet(dirname(argv$sample_sheet))
  colnames(targets)[1] <- "Sample_Name"
  missing_s <- targets %>% filter(!str_detect(Basename, "/")) %>% pull(Sample_Name)
  if (length(missing_s)) message(paste0("Samples ", paste(missing_s, collapse = ", "), " have no IDAT data"))
  targets <- targets %>% filter(str_detect(Basename, "/"))
  rgSet <- read.metharray.exp(targets = targets)
  if (argv$hg_build == 38 && rgSet@annotation["array"] == "IlluminaHumanMethylationEPIC")
    rgSet@annotation["annotation"] <- "ilm10b5.hg38"

  detP <- detectionP(rgSet)
  keep <- colMeans(detP) < argv$samples_pval_cutoff
  rgSet <- rgSet[, keep]; targets <- targets[keep, ]
  mSetSq <- preprocessQuantile(rgSet)
  mSetSq <- mSetSq[!(featureNames(mSetSq) %in% cross_reactive), ]
  if (argv$keep_only_cpg_probes) mSetSq <- dropLociWithSnps(mSetSq)
  detP <- detP[match(featureNames(mSetSq), rownames(detP)), ]
  keep <- rowSums(detP < argv$probe_pval_cutoff) == ncol(mSetSq)
  mSetSq <- mSetSq[keep, ]

  cd <- as.data.frame(rgSet@colData)
  to_name <- setNames(as.character(cd$Sample_Name), rownames(cd))
  ren <- function(x) ifelse(x %in% names(to_name), to_name[x], x)
  bval <- getBeta(mSetSq) %>% as_tibble(rownames = "ID") %>% rename_with(ren)
  mval <- getM(mSetSq)    %>% as_tibble(rownames = "ID") %>% rename_with(ren)
  vroom::vroom_write(bval, argv$output_beta, delim = "\t")
  vroom::vroom_write(mval, argv$output_m, delim = "\t")
  saveRDS(list(rgSet = rgSet, mSetSq = "mSetSq", mSetSqbval = "mSetSqbval"), argv$output_rds)
  message("minfi analysis completed!")
}

# ---------------------------------------------------------------------------
annotate_step <- function(argv) {
  suppressPackageStartupMessages({ library(sesame); library(Rsamtools); library(dplyr) })
  sesameData::sesameDataCache()
  betas <- vroom::vroom(argv$input_beta, delim = "\t", show_col_types = FALSE)
  Mv    <- vroom::vroom(argv$input_m, delim = "\t", show_col_types = FALSE)
  pa <- sesameData::sesameData_annoProbes(betas$ID, column = "gene_id")
  probe_annot <- data.frame(ID = names(pa), as.data.frame(pa), check.names = FALSE, row.names = NULL)
  loc <- probe_annot %>% transmute(chr = as.character(seqnames), start, end, ID)

  build_bed <- function(df) {
    inner_join(loc, df, by = "ID") %>%
      mutate(end = start + 1, chr_num = as.numeric(stringr::str_remove(chr, "chr"))) %>%
      arrange(chr_num, chr, start) %>% select(-chr_num)
  }
  write_bed_gz <- function(df, out_gz) {
    plain <- sub("\\.gz$", "", out_gz)
    names(df)[1] <- "#chr"
    dir.create(dirname(out_gz), showWarnings = FALSE, recursive = TRUE)
    vroom::vroom_write(df, plain, delim = "\t")
    Rsamtools::bgzip(plain, dest = out_gz, overwrite = TRUE); file.remove(plain)
    Rsamtools::indexTabix(out_gz, format = "bed")
  }
  write_bed_gz(build_bed(betas), argv$output_beta_bed)
  write_bed_gz(build_bed(Mv), argv$output_m_bed)
  vroom::vroom_write(probe_annot, argv$output_annot, delim = "\t")
  message("methylation annotation completed!")
}

# ---------------------------------------------------------------------------
switch(argv$step,
  sesame   = sesame_step(argv),
  minfi    = minfi_step(argv),
  annotate = annotate_step(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
