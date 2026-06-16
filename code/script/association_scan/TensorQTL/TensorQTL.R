#!/usr/bin/env Rscript
# TensorQTL.R — R helper for TensorQTL.py
#
# Usage:
#   Rscript TensorQTL.R nominal_qvalues <tsv> [<interaction>]
#   Rscript TensorQTL.R trans_qvalues <tsv>
#   Rscript TensorQTL.R regional_postprocess <out_tsv.gz> <out_summary.txt> <file1.tsv.gz> [...]

suppressPackageStartupMessages(library(data.table))

# Storey q-values with BH fallback
safe_qvalue <- function(p) {
  p[is.na(p)] <- 1
  tryCatch(
    suppressWarnings(qvalue::qvalue(p, fdr.level = 0.05)$qvalues),
    error = function(e) p.adjust(p, method = "BH")
  )
}

# Resolve interaction variable name from either a file path or a bare name
get_interaction_name <- function(interaction_arg) {
  if (!nzchar(interaction_arg)) return("")
  if (file.exists(interaction_arg)) {
    hdr <- colnames(fread(interaction_arg, nrows = 0))
    return(if (length(hdr) >= 2) hdr[2L] else "")
  }
  interaction_arg
}

# nominal_qvalues: add qvalue per gene; interaction mode adds qvalue_main + qvalue_interaction
cmd_nominal_qvalues <- function(tsv_path, interaction_arg = "") {
  dt <- fread(tsv_path, sep = "\t")
  interaction_name <- get_interaction_name(interaction_arg)

  if (!nzchar(interaction_name)) {
    dt[, qvalue := safe_qvalue(pvalue), by = molecular_trait_id]
  } else {
    int_col <- paste0("pvalue_", interaction_name, "_interaction")
    dt[, qvalue_main := safe_qvalue(pvalue), by = molecular_trait_id]
    if (int_col %in% colnames(dt)) {
      dt[, qvalue_interaction := safe_qvalue(get(int_col)), by = molecular_trait_id]
    }
  }

  fwrite(dt, tsv_path, sep = "\t", quote = FALSE, na = "NA")
  message("nominal_qvalues done: ", tsv_path)
}

# trans_qvalues: add global qvalue across all trans pairs
cmd_trans_qvalues <- function(tsv_path) {
  dt <- fread(tsv_path, sep = "\t")
  if (nrow(dt) > 0L) {
    dt[, qvalue := safe_qvalue(pvalue)]
  } else {
    dt[, qvalue := numeric(0L)]
  }
  fwrite(dt, tsv_path, sep = "\t", quote = FALSE, na = "NA")
  message("trans_qvalues done: ", tsv_path)
}

# regional_postprocess: merge per-chromosome regional files, add FDR/q-value columns
cmd_regional_postprocess <- function(out_tsv, out_summary, regional_files) {
  dts <- lapply(regional_files, function(f) {
    tryCatch(fread(f, sep = "\t"), error = function(e) {
      warning("Skipping unreadable file: ", f, " (", conditionMessage(e), ")")
      NULL
    })
  })
  dts <- Filter(Negate(is.null), dts)
  if (length(dts) == 0L) stop("No valid regional files found")

  dt <- rbindlist(dts, fill = TRUE)

  # Sort by chromosome then position
  dt[, .chrom_order := suppressWarnings(as.integer(chrom))]
  setorder(dt, .chrom_order, pos)
  dt[, .chrom_order := NULL]

  dt[, fdr_beta := p.adjust(p_beta, method = "BH")]
  dt[, fdr_perm := p.adjust(p_perm, method = "BH")]
  dt[, q_beta   := safe_qvalue(p_beta)]
  dt[, q_perm   := safe_qvalue(p_perm)]

  out_tsv_raw <- sub("\\.gz$", "", out_tsv)
  fwrite(dt, out_tsv_raw, sep = "\t", quote = FALSE, na = "NA")
  system2("bgzip", c("--compress-level", "9", "-f", out_tsv_raw))
  system2("tabix", c("-S", "1", "-s", "1", "-b", "2", "-e", "2", out_tsv))

  n_tested  <- nrow(dt)
  n_objects <- uniqueN(dt$molecular_trait_object_id)
  n_chroms  <- uniqueN(dt$chrom)
  writeLines(c(
    "TensorQTL cis-QTL regional significance summary",
    paste0("Input files:              ", length(regional_files)),
    paste0("Chromosomes:              ", n_chroms),
    paste0("Molecular traits tested:  ", n_tested),
    paste0("Molecular trait objects:  ", n_objects),
    "",
    "Significance thresholds (p_beta):",
    paste0("  q_beta  < 0.05: ", sum(dt$q_beta  < 0.05, na.rm = TRUE)),
    paste0("  q_beta  < 0.10: ", sum(dt$q_beta  < 0.10, na.rm = TRUE)),
    paste0("  fdr_beta < 0.05: ", sum(dt$fdr_beta < 0.05, na.rm = TRUE)),
    paste0("  fdr_beta < 0.10: ", sum(dt$fdr_beta < 0.10, na.rm = TRUE))
  ), out_summary)

  message("regional_postprocess done: ", out_tsv)
  message("Summary: ", out_summary)
}

# Dispatch
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("Usage: TensorQTL.R <subcommand> <args...>")

subcmd <- args[1L]
if (subcmd == "nominal_qvalues") {
  if (length(args) < 2L) stop("nominal_qvalues requires <tsv_path>")
  cmd_nominal_qvalues(args[2L], if (length(args) >= 3L) args[3L] else "")
} else if (subcmd == "trans_qvalues") {
  if (length(args) < 2L) stop("trans_qvalues requires <tsv_path>")
  cmd_trans_qvalues(args[2L])
} else if (subcmd == "regional_postprocess") {
  if (length(args) < 4L) stop("regional_postprocess requires <out_tsv> <out_summary> <file1> ...")
  cmd_regional_postprocess(args[2L], args[3L], args[-(1:3)])
} else {
  stop(paste("Unknown subcommand:", subcmd))
}
