# manifest_common.R
#
# Shared helpers for the pecotmr_integration MANIFEST wrapper scripts
# (region_manifest / gwas_rss_manifest / ctwas_manifest / twas_manifest /
# enloc_manifest / colocboost_manifest / mash_manifest / ld_blocks_to_manifest).
# Each script sources this from its own directory:
#
#   .d <- dirname(sub("^--file=", "",
#           grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
#   source(file.path(.d, "manifest_common.R"))
#
# These are I/O + orchestration utilities only (no analysis logic -- that lives
# in pecotmr). All table I/O goes through readr for consistency; the reader
# wrappers reproduce the base-R read.table semantics the scripts relied on
# (all-character columns, `#chr` header kept verbatim, only the literal "NA"
# treated as missing, no whitespace trimming, `#`-comment handling per caller).

suppressPackageStartupMessages(library(readr))

# ---- table I/O --------------------------------------------------------------

# Header TSV -> data.frame, every column character, column names kept verbatim
# (so `#chr` survives). Matches read.table(header=TRUE, sep="\t", quote="",
# comment.char="", check.names=FALSE, stringsAsFactors=FALSE, na.strings="NA",
# strip.white=FALSE).
readMeta <- function(path)
  as.data.frame(
    readr::read_delim(path, delim = "\t", quote = "", comment = "", na = "NA",
                      col_types = readr::cols(.default = readr::col_character()),
                      name_repair = "minimal", trim_ws = FALSE, progress = FALSE,
                      show_col_types = FALSE),
    stringsAsFactors = FALSE, check.names = FALSE)

# Headerless whitespace/TSV table (e.g. a BED of association windows, or a
# region-list) -> data.frame, character columns, `#`-prefixed lines dropped as
# comments. Matches read.table(header=FALSE, comment.char="#") for both
# whitespace- and tab-delimited inputs (runs of whitespace collapse to one
# delimiter, as read_table does).
readTableNoHeader <- function(path)
  as.data.frame(
    readr::read_table(path, col_names = FALSE, comment = "#", na = "NA",
                      col_types = readr::cols(.default = readr::col_character()),
                      progress = FALSE, show_col_types = FALSE),
    stringsAsFactors = FALSE)

# Write a manifest data.frame as a TSV. Matches write.table(sep="\t",
# quote=FALSE, row.names=FALSE, na="") + creates the parent dir.
writeManifest <- function(df, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  readr::write_tsv(df, path, na = "", quote = "none", progress = FALSE)
}

# ---- chromosome normalization (two opposite conventions in use) -------------
# add a "chr" prefix when missing ("22" -> "chr22"; "chr22" kept)
chromAdd <- function(x) {
  x <- as.character(x)
  ifelse(is.na(x) | !nzchar(x), x,
         ifelse(startsWith(x, "chr"), x, paste0("chr", x)))
}
# strip a leading "chr" ("chr22" -> "22")
chromStrip <- function(x) sub("^chr", "", as.character(x), ignore.case = TRUE)

# ---- comma-list helpers -----------------------------------------------------
splitC     <- function(x) { v <- trimws(strsplit(as.character(x), ",")[[1L]]); v[nzchar(v)] }
joinC      <- function(v) paste(v, collapse = ",")
makeUnique <- function(x) joinC(unique(splitC(x)))

# ---- path helpers -----------------------------------------------------------
# Prefix each comma-separated file with `pre` (skip when pre is empty).
prefixPaths <- function(x, pre)
  joinC(vapply(splitC(x),
               function(f) if (nzchar(pre)) file.path(pre, f) else f, character(1)))

# Resolve a possibly-stale path against the meta file that referenced it:
# exists as given -> basename in cwd -> basename in meta dir -> meta_dir/path.
adaptFilePath <- function(filePath, referenceFile) {
  filePath <- trimws(filePath)
  refDir <- dirname(referenceFile)
  isFile <- function(f) file.exists(f) && !dir.exists(f)
  if (isFile(filePath)) return(filePath)
  fileName <- basename(filePath)
  if (isFile(fileName)) return(fileName)
  inRef <- file.path(refDir, fileName)
  if (isFile(inRef)) return(inRef)
  prefixed <- file.path(refDir, filePath)
  if (isFile(prefixed)) return(prefixed)
  stop("No valid path found for file: ", filePath)
}
