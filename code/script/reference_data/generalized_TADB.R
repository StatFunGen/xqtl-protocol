#!/usr/bin/env Rscript
# ============================================================
# generalized_TADB.R
# Standalone CLI worker for generalized_TADB.ipynb [default].
#
# From brain/tissue TAD coordinates + a gene-coordinate table, build:
#   generalized_TAD.tsv   — TADs after recursive overlap-merge
#   generalized_TADB.tsv  — TAD-boundary (TADB) windows
#   TADB_enhanced_cis.bed — per-gene TADB-enhanced cis windows
#   extended_TADB.bed     — extended TADBs
#
# Conventions: read with vroom, manipulate with dplyr/purrr (no data.table).
# CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(purrr)
  library(tibble)
  library(dplyr)   # last
})

parser <- arg_parser("generalized TADB builder")
parser <- add_argument(parser, "--tad-input", type = "character",
                       help = "Brain/tissue TAD coordinates (chr start end, no header)")
parser <- add_argument(parser, "--gene-coords", type = "character",
                       help = "Gene coordinates table (index, #chr, start, end, gene_id, gene_name)")
parser <- add_argument(parser, "--overlap-cutoff", type = "numeric", default = 80,
                       help = "Min percent overlap to merge two TADs into one generalized TAD")
parser <- add_argument(parser, "--output-gentad", type = "character", help = "generalized_TAD.tsv")
parser <- add_argument(parser, "--output-gentadb", type = "character", help = "generalized_TADB.tsv")
parser <- add_argument(parser, "--output-cis", type = "character", help = "TADB_enhanced_cis.bed")
parser <- add_argument(parser, "--output-ext", type = "character", help = "extended_TADB.bed")
argv <- parse_args(parser)
for (f in c("tad_input", "gene_coords", "output_gentad", "output_gentadb", "output_cis", "output_ext"))
  if (is.na(argv[[f]])) stop(sprintf("--%s is required", gsub("_", "-", f)))
overlap_cutoff <- argv$overlap_cutoff
for (p in unique(dirname(c(argv$output_gentad, argv$output_gentadb, argv$output_cis, argv$output_ext))))
  dir.create(p, showWarnings = FALSE, recursive = TRUE)

find_TAD_overlap <- function(x, inputDF) {
  rowChr <- x['chr']; rowStart <- as.numeric(x['start']); rowEnd <- as.numeric(x['end'])
  rowTADIndex <- x['TAD_index']
  TADsubset <- inputDF %>% filter(chr == rowChr)
  TADsubset$start <- as.numeric(TADsubset$start); TADsubset$end <- as.numeric(TADsubset$end)
  priorTADsubset <- TADsubset %>%
    filter(start <= rowStart & rowStart <= end & (start != rowStart | end != rowEnd)) %>% arrange(start)
  nextTADsubset <- TADsubset %>%
    filter(start <= rowEnd & rowEnd <= end & (start != rowStart | end != rowEnd)) %>% arrange(-end)
  completeOverlapSubset <- TADsubset %>%
    filter(start <= rowStart & rowEnd <= end & (start != rowStart | end != rowEnd))
  priorOverlap <- 0; prior_TAD_index <- rowTADIndex
  nextOverlap <- 0; next_TAD_index <- rowTADIndex; completeOverlap <- FALSE
  if (nrow(priorTADsubset)) {
    priorOverlap <- priorTADsubset %>%
      mutate(inner_TAD_Length = end - rowStart, outer_TAD_Length = end - start,
             overlap = (inner_TAD_Length / outer_TAD_Length) * 100) %>% arrange(-overlap)
    prior_TAD_index <- priorOverlap$TAD_index[1]; priorOverlap <- priorOverlap$overlap[1]
    if (is.na(prior_TAD_index)) stop(paste("The following TAD has an issue:", rowTADIndex))
  }
  if (nrow(nextTADsubset)) {
    nextOverlap <- nextTADsubset %>%
      mutate(inner_TAD_Length = rowEnd - start, outer_TAD_Length = end - start,
             overlap = (inner_TAD_Length / outer_TAD_Length) * 100) %>% arrange(-overlap)
    next_TAD_index <- nextOverlap$TAD_index[1]; nextOverlap <- nextOverlap$overlap[1]
    if (is.na(next_TAD_index)) stop(paste("The following TAD has an issue:", rowTADIndex))
  }
  if (nrow(completeOverlapSubset)) completeOverlap <- TRUE
  list(prior = as.character(priorOverlap), subsequent = as.character(nextOverlap),
       com = as.character(completeOverlap), prior_tad = as.character(prior_TAD_index),
       next_tad = as.character(next_TAD_index))
}

merge_TADs <- function(x, inputDF, cutoff = overlap_cutoff) {
  rowStart <- as.numeric(x["start"]); rowEnd <- as.numeric(x["end"])
  rowTADIndex <- as.character(x['TAD_index'])
  rowPriorOverlap <- as.double(x["prior_overlap"]); rowPriorTADIndex <- as.character(x["prior_TAD_index"])
  rowNextOverlap <- as.double(x["next_overlap"]); rowNextTADIndex <- as.character(x["next_TAD_index"])
  newStart <- rowStart; newEnd <- rowEnd
  if (rowNextOverlap >= cutoff && rowNextTADIndex != rowTADIndex) {
    newEnd <- inputDF %>% filter(TAD_index == rowNextTADIndex); newEnd <- newEnd$end[1]
  }
  if (rowPriorOverlap >= cutoff & rowPriorTADIndex != rowTADIndex) {
    newStart <- inputDF %>% filter(TAD_index == rowPriorTADIndex); newStart <- newStart$start[1]
  }
  list(newstart = newStart, newend = newEnd)
}

annotate_overlaps <- function(df) {
  df$TAD_index <- paste0('TAD_', seq_len(nrow(df)))
  res <- apply(df, 1, find_TAD_overlap, df)
  df$prior_overlap    <- as.double(lapply(res, "[[", 'prior'))
  df$prior_TAD_index  <- as.character(lapply(res, "[[", 'prior_tad'))
  df$next_overlap     <- as.double(lapply(res, "[[", 'subsequent'))
  df$next_TAD_index   <- as.character(lapply(res, "[[", 'next_tad'))
  df$complete_overlap <- as.logical(lapply(res, "[[", 'com'))
  df
}

recursive_merge <- function(tadDF) {
  tadDF <- annotate_overlaps(tadDF)
  merge_results <- apply(tadDF, 1, merge_TADs, tadDF, overlap_cutoff)
  tadDF$end   <- as.numeric(lapply(merge_results, "[[", 'newend'))
  tadDF$start <- as.numeric(lapply(merge_results, "[[", 'newstart'))
  candidateDF <- tadDF %>% distinct(chr, start, end, .keep_all = TRUE)
  if (nrow(tadDF) == nrow(candidateDF)) return(candidateDF)
  candidateDF <- annotate_overlaps(candidateDF)
  candidateDF <- candidateDF %>% filter(complete_overlap == FALSE)
  candidateDF <- annotate_overlaps(candidateDF)
  recursive_merge(candidateDF)
}

# ---- Step i. Manage TAD redundancy ----
general_TAD_DF <- vroom::vroom(argv$tad_input, delim = "\t",
                               col_names = c('chr', 'start', 'end'), show_col_types = FALSE)
general_TAD_DF <- general_TAD_DF[with(general_TAD_DF, order(chr, start, -end)), ]
final_brain_TAD_DF <- recursive_merge(general_TAD_DF)
formatted_final_DF <- final_brain_TAD_DF %>% subset(select = c("chr", "start", "end"))
formatted_final_DF$start <- format(formatted_final_DF$start, scientific = FALSE)
formatted_final_DF$end   <- format(formatted_final_DF$end, scientific = FALSE)
vroom::vroom_write(formatted_final_DF, argv$output_gentad, delim = "\t", col_names = FALSE)

# ---- Step ii. Generalized TADB windows ----
general  <- vroom::vroom(argv$output_gentad, delim = "\t",
                         col_names = c("chr", "start", "end"), show_col_types = FALSE) %>%
  mutate(row_index = row_number())
specific <- vroom::vroom(argv$tad_input, delim = "\t",
                         col_names = c("chr", "start", "end"), show_col_types = FALSE) %>%
  mutate(row_index = row_number())

chromosomes <- c(paste0('chr', seq(1, 22)), "chrX")
bplengths <- c(248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636,
               138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345,
               83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895)
chrDF <- data.frame(chr = chromosomes, left = bplengths)

find_left_boundary <- function(chrm, start_pos, end_pos) {
  within_TAD <- specific %>% filter(chr == chrm & start >= start_pos & end <= end_pos)
  if (nrow(within_TAD) == 1) within_TAD %>% pull(start) %>% as.integer()
  else (within_TAD %>% pull(start) %>% sort())[2] %>% as.integer()
}
left_table <- general %>%
  mutate(left = pmap(list(chrm = chr, start_pos = start, end_pos = end), find_left_boundary)) %>%
  select(chr, left) %>% mutate(left = as.integer(left))
left_table_final <- rbind(left_table, chrDF)

find_right_boundary <- function(chrm, start_pos, end_pos) {
  within_TAD <- specific %>% filter(chr == chrm & start >= start_pos & end <= end_pos)
  if (nrow(within_TAD) == 1) within_TAD %>% pull(end) %>% as.integer()
  else (within_TAD %>% pull(end) %>% sort(decreasing = TRUE))[2] %>% as.integer()
}
right_table <- general %>%
  mutate(right = pmap(list(chrm = chr, start_pos = start, end_pos = end), find_right_boundary)) %>%
  select(chr, right) %>% mutate(right = as.integer(right))
right_table_final <- rbind(right_table, tibble(chr = chromosomes, right = 0))

find_next_boundary <- function(chrm, end_pos)
  (left_table_final %>% filter(chr == chrm, left >= end_pos) %>% pull(left) %>% sort())[1]
find_previous_boundary <- function(chrm, start_pos)
  (right_table_final %>% filter(chr == chrm, right <= start_pos) %>% pull(right) %>% sort(decreasing = TRUE))[1]

TADB <- general %>%
  mutate(previous_bound = pmap(list(chr, start), find_previous_boundary),
         next_bound = pmap(list(chr, end), find_next_boundary)) %>%
  select(-start, -end) %>% rename(start = previous_bound) %>% rename(end = next_bound) %>%
  mutate(start = as.integer(start), end = as.integer(end)) %>% select(chr, start, end) %>% distinct()

if_fully_cover <- function(chrm, start_pos, end_pos) {
  potential <- TADB %>% filter(chr == chrm, start <= start_pos, end >= end_pos)
  if (nrow(potential) <= 1) 0 else 1
}
TADB_final <- TADB %>%
  mutate(cover_status = pmap(list(chrm = chr, start_pos = start, end_pos = end), if_fully_cover)) %>%
  filter(cover_status != 1) %>% mutate(index = paste0("TADB", row_number())) %>%
  select(-cover_status) %>% rename(`#chr` = chr)
vroom::vroom_write(TADB_final, argv$output_gentadb, delim = "\t")

# ---- gene cis windows ----
all_gene <- vroom::vroom(argv$gene_coords, delim = "\t", show_col_types = FALSE)
all_gene <- all_gene[, c("#chr", "start", "end", "gene_id")]        # drop the index col + gene_name
names(all_gene) <- c("chr", "gene_start", "gene_end", "gene_id")
all_gene <- distinct(all_gene)
TADB_final <- TADB_final %>% rename(chr = "#chr")
gene_TADB_tb <- left_join(all_gene, TADB_final, by = "chr", relationship = "many-to-many") %>%
  filter(!(gene_start > end) & !(gene_end < start))

ordered_chr <- c(paste0("chr", as.character(1:22)), "chrX")
cis_TADB <- left_join(gene_TADB_tb, chrDF, by = "chr") %>% rename(chr_end = left) %>%
  mutate(cis_start = pmax(0, gene_start - 1000000), cis_end = pmin(chr_end, gene_end + 1000000))
extended <- cis_TADB %>%
  mutate(true_start = pmin(start, cis_start), true_end = pmax(end, cis_end)) %>%
  arrange(chr, true_start) %>% ungroup() %>%
  select(chr, gene_id, true_start, true_end) %>% rename(start = true_start, end = true_end)
extended_final <- extended %>% group_by(chr, gene_id) %>%
  summarize(start = min(start), end = max(end), .groups = "drop") %>% arrange(chr, start)
extend_cis_ordered <- extended_final %>% mutate(chr = factor(chr, levels = ordered_chr)) %>%
  arrange(chr) %>% rename(`#chr` = chr) %>% select(`#chr`, start, end, gene_id)
vroom::vroom_write(extend_cis_ordered, argv$output_cis, delim = "\t")

# ---- extended TADB ----
cis_summary <- cis_TADB %>% group_by(index) %>%
  summarize(min_start = min(cis_start), max_end = max(cis_end), .groups = "drop")
extend_TADB <- left_join(cis_summary, TADB_final, by = "index") %>%
  mutate(new_start = pmin(start, min_start), new_end = pmax(end, max_end)) %>%
  select(chr, new_start, new_end) %>% rename(start = new_start, end = new_end) %>%
  mutate(index = paste0("TADB", row_number())) %>%
  mutate(chr = factor(chr, levels = ordered_chr)) %>% arrange(chr, start) %>%
  rename(`#chr` = chr) %>% mutate(index = paste0("TADB_", row_number()))
vroom::vroom_write(extend_TADB, argv$output_ext, delim = "\t")

cat(sprintf("DONE: generalized TADs = %d | TADB_final = %d | cis windows = %d | extended TADB = %d\n",
            nrow(formatted_final_DF), nrow(TADB_final), nrow(extend_cis_ordered), nrow(extend_TADB)))
