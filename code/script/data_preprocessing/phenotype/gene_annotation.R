#!/usr/bin/env Rscript
# ============================================================
# gene_annotation.R
# Standalone CLI worker for gene_annotation.ipynb steps.
#
# Steps (selected via --step):
#   annotate_coord               — prepend gene/protein/atac coordinates to a matrix
#   map_leafcutter_cluster_to_gene — map leafcutter intron clusters to genes
#   annotate_leafcutter_isoforms — coordinate-annotate leafcutter introns per gene
#   annotate_psichomics_isoforms — coordinate-annotate psichomics events per gene
#   annotate_coord_biomart       — fetch coordinates from Ensembl biomaRt (network)
#
# Ports the Broad `qtl`/`pyqtl` logic to rtracklayer + GenomicRanges + Rsamtools.
# Conventions: read with vroom, manipulate with dplyr/tidyr, bgzip/tabix via
# Rsamtools (no data.table). CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(rtracklayer)
  library(GenomicRanges)
  library(Rsamtools)
  library(tidyr)
  library(dplyr)   # last, so dplyr verbs win over S4Vectors::rename() etc.
})

# ---------------------------------------------------------------------------
# Argument parsing (flags mirror gene_annotation.py / the notebook CLI)
# ---------------------------------------------------------------------------
parser <- arg_parser("gene_annotation worker (see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "Step to run")
parser <- add_argument(parser, "--phenoFile", type = "character", default = "",
                       help = "Molecular phenotype matrix")
parser <- add_argument(parser, "--coordinate-annotation", type = "character", default = "",
                       help = "[annotate_coord] collapsed gene-model GTF (or ATAC index)")
parser <- add_argument(parser, "--annotation-gtf", type = "character", default = "",
                       help = "[leafcutter/psichomics] full gene-model GTF with exons")
parser <- add_argument(parser, "--sample-participant-lookup", type = "character", default = "",
                       help = "Optional sample_id -> participant_id map")
parser <- add_argument(parser, "--molecular-trait-type", type = "character", default = "gene",
                       help = "gene | protein | atac")
parser <- add_argument(parser, "--phenotype-id-column", type = "character", default = "gene_id",
                       help = "gene_id or gene_name")
parser <- add_argument(parser, "--auxiliary-id-mapping", type = "character", default = "",
                       help = "[protein] optional protein-name index")
parser <- add_argument(parser, "--strip-id", flag = TRUE, help = "Strip 'x|' prefix from feature IDs")
parser <- add_argument(parser, "--sep", type = "character", default = "\t",
                       help = "Input field delimiter")
parser <- add_argument(parser, "--output-bed", type = "character", default = "",
                       help = "Output coordinate-annotated BED.gz")
parser <- add_argument(parser, "--output-region-list", type = "character", default = "",
                       help = "Output region-list TSV")
parser <- add_argument(parser, "--gene-list-output", type = "character", default = "",
                       help = "[annotate_coord] output gene-list TSV")
parser <- add_argument(parser, "--intron-count", type = "character", default = "",
                       help = "[map_leafcutter] leafcutter intron-count table")
parser <- add_argument(parser, "--map-stra", type = "character", default = "site",
                       help = "[map_leafcutter] mapping strategy: site | region")
parser <- add_argument(parser, "--overlap-ratio", type = "numeric", default = 0.8,
                       help = "[map_leafcutter region] min gene-overlap fraction")
parser <- add_argument(parser, "--output-exon-list", type = "character", default = "",
                       help = "[map_leafcutter] output exon list")
parser <- add_argument(parser, "--output-cluster-map", type = "character", default = "",
                       help = "[map_leafcutter] output cluster-to-gene map")
parser <- add_argument(parser, "--cluster-map", type = "character", default = "",
                       help = "[annotate_leafcutter] input cluster-to-gene map")
parser <- add_argument(parser, "--output-phenotype-group", type = "character", default = "",
                       help = "[leafcutter/psichomics] output phenotype-group file")
parser <- add_argument(parser, "--ensembl-version", type = "numeric", default = NA,
                       help = "[annotate_coord_biomart] Ensembl release version")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
have <- function(x) !is.null(x) && length(x) == 1 && !is.na(x) && nzchar(x)

# Write a BED data frame as bgzipped + tabix-indexed (first col -> "#chr").
write_bed_r <- function(df, path) {
  names(df)[1] <- "#chr"
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  plain <- sub("\\.gz$", "", path)
  vroom::vroom_write(df, plain, delim = "\t")
  Rsamtools::bgzip(plain, dest = path, overwrite = TRUE)
  file.remove(plain)
  Rsamtools::indexTabix(path, format = "bed")
}

# Per-gene coordinate template from a collapsed GTF (qtl uses 0-based start-1/end-1).
build_template <- function(gr) {
  genes <- gr[gr$type == "gene"]
  if (any(duplicated(genes$gene_id)))
    stop("coordinate_annotation must be collapsed into a gene model (duplicate gene rows)")
  tx <- gr[gr$type == "transcript"]
  tibble(chr = as.character(seqnames(tx)),
         start = start(tx) - 1L, end = end(tx) - 1L,
         strand = as.character(strand(tx)),
         gene_id = tx$gene_id, gene_name = tx$gene_name) %>%
    group_by(gene_id) %>%
    summarise(chr = dplyr::first(chr), start = min(start), end = max(end),
              strand = dplyr::first(strand), gene_name = dplyr::first(gene_name),
              .groups = "drop") %>%
    select(chr, start, end, gene_id, strand, gene_name)
}

# TSS BED (0-based, strand-aware) keyed by gene_id or gene_name.
tss_bed <- function(gr, feature = "gene", phenotype_id = "gene_id") {
  sub <- gr[gr$type == feature]
  plus <- as.character(strand(sub)) == "+"
  idv <- if (phenotype_id == "gene_name") sub$gene_name else sub$gene_id
  tibble(chr = as.character(seqnames(sub)),
         start = ifelse(plus, start(sub) - 1L, end(sub) - 1L),
         end   = ifelse(plus, start(sub), end(sub)),
         gene_id = idv) %>%
    arrange(chr, start)
}

# First-transcript exon table (mirrors qtl.annotation, 1-based coordinates).
exon_table_from_gtf <- function(gr) {
  ex <- gr[gr$type == "exon"]
  d <- tibble(chr = as.character(seqnames(ex)), start = start(ex), end = end(ex),
              strand = as.character(strand(ex)), gene_id = ex$gene_id,
              gene_name = ex$gene_name, transcript_id = ex$transcript_id)
  first_tx <- d %>% distinct(gene_id, transcript_id) %>%
    group_by(gene_id) %>% slice(1) %>% ungroup()
  d %>% semi_join(first_tx, by = c("gene_id", "transcript_id")) %>%
    select(chr, start, end, strand, gene_id, gene_name)
}

read_matrix <- function(path, sep) {
  drop_cols <- c("#chr", "chr", "start", "end", "stop",
                 "annot.seqnames", "annot.start", "annot.end")
  vroom::vroom(path, delim = sep, col_types = cols(.default = "c"),
               show_col_types = FALSE) %>% select(-any_of(drop_cols))
}

rename_samples_lookup <- function(df, lookup) {
  if (!have(lookup) || !file.exists(lookup) || dir.exists(lookup)) return(df)
  sep <- if (grepl("\\.csv$", lookup)) "," else "\t"
  lk <- tryCatch(vroom::vroom(lookup, delim = sep, col_types = cols(.default = "c"),
                              show_col_types = FALSE), error = function(e) NULL)
  if (is.null(lk)) return(df)
  m <- if ("genotype_id" %in% names(lk) && ncol(lk) >= 2)
         setNames(lk$genotype_id, lk[[2]]) else setNames(lk[[2]], lk[[1]])
  nm <- names(df); hit <- nm %in% names(m); nm[hit] <- unname(m[nm[hit]]); names(df) <- nm
  df
}

read_participant_map <- function(lookup) {
  if (!have(lookup) || !file.exists(lookup) || dir.exists(lookup)) return(NULL)
  lk <- vroom::vroom(lookup, delim = "\t", col_types = cols(.default = "c"),
                     show_col_types = FALSE)
  if (!("participant_id" %in% names(lk))) return(NULL)
  setNames(lk$participant_id, lk[[1]])
}

apply_participant_map <- function(df, pmap, first_sample_col = 5) {
  if (is.null(pmap)) return(df)
  nm <- names(df); idx <- first_sample_col:length(nm)
  hit <- nm[idx] %in% names(pmap); nm[idx][hit] <- unname(pmap[nm[idx][hit]])
  names(df) <- nm; df
}

# ---------------------------------------------------------------------------
# Step: annotate_coord
# ---------------------------------------------------------------------------
annotate_coord <- function(argv) {
  mtt <- argv$molecular_trait_type
  df <- read_matrix(argv$phenoFile, argv$sep)
  id_name <- names(df)[1]
  df <- rename_samples_lookup(df, argv$sample_participant_lookup)
  sample_cols <- setdiff(names(df), id_name)

  region_cols_of <- function(bed) intersect(c("#chr", "start", "end", "ID", "strand"), names(bed))
  write_region_list <- function(bed) {
    rl <- bed %>% select(all_of(region_cols_of(bed))) %>% mutate(path = argv$output_bed)
    vroom::vroom_write(rl, argv$output_region_list, delim = "\t")
  }

  if (mtt == "gene") {
    if (argv$strip_id)
      df[[id_name]] <- ifelse(grepl("\\|", df[[id_name]]), sub(".*\\|", "", df[[id_name]]), df[[id_name]])
    gr <- rtracklayer::import(argv$coordinate_annotation)
    tmpl <- build_template(gr)
    key <- argv$phenotype_id_column
    dfj <- df; names(dfj)[1] <- ".join"
    merged <- dfj %>%
      inner_join(tmpl %>% mutate(.join = .data[[key]]), by = ".join") %>%
      add_count(gene_id, name = ".n") %>% filter(.data$.n == 1) %>% select(-.n)
    bed <- merged %>% mutate(ID = gene_id) %>%
      select(`#chr` = chr, start, end, ID, strand, all_of(sample_cols)) %>%
      arrange(`#chr`, start, end)
    write_bed_r(bed, argv$output_bed)
    write_region_list(bed)
    gl_out <- if (have(argv$gene_list_output)) argv$gene_list_output else
      file.path(dirname(argv$output_bed),
                paste0(sub("\\.gz$", "", basename(argv$phenoFile)), ".gene_list.tsv"))
    gl <- tmpl %>% filter(.data[[key]] %in% df[[1]]) %>% rename(`#chr` = chr)
    vroom::vroom_write(gl, gl_out, delim = "\t")

  } else if (mtt == "protein") {
    gr <- rtracklayer::import(argv$coordinate_annotation)
    tmpl <- build_template(gr)
    key <- argv$phenotype_id_column
    aux <- argv$auxiliary_id_mapping
    if (have(aux) && file.exists(aux) && !dir.exists(aux)) {
      # SOMAscan-style: aux maps the matrix ID column -> EntrezGeneSymbol (gene_name)
      # + UniProt. Ported from gene_annotation.py for parity; NOT covered by a fixture
      # (the MWE has no SOMAscan protein data), so this path is unvalidated.
      info <- vroom::vroom(aux, show_col_types = FALSE)
      names(info)[names(info) == "EntrezGeneSymbol"] <- "gene_name"
      info <- info %>% select(all_of(c("gene_name", id_name, "UniProt")))
      sp <- df %>% inner_join(info, by = id_name) %>%
        rename(.gname = gene_name, .uniprot = UniProt) %>%
        select(-all_of(id_name))
    } else {
      dfj <- df; names(dfj)[1] <- ".gid"
      sp <- dfj %>% separate(.gid, into = c(".gname", ".uniprot"),
                             sep = "\\|", extra = "merge", fill = "right")
    }
    merged <- sp %>%
      inner_join(tmpl %>% mutate(.join = .data[[key]]), by = c(".gname" = ".join")) %>%
      mutate(ID = paste0(gene_id, "_", .uniprot)) %>%
      add_count(ID, name = ".n") %>% filter(.data$.n == 1) %>% select(-.n) %>%
      arrange(chr, start, end)
    bed <- bind_cols(
      merged %>% transmute(`#chr` = chr, start, end, ID, gene_name = .gname),
      merged %>% select(all_of(sample_cols)))
    write_bed_r(bed, argv$output_bed)
    write_region_list(bed)
    gl_out <- if (have(argv$gene_list_output)) argv$gene_list_output else
      file.path(dirname(argv$output_bed),
                paste0(sub("\\.gz$", "", basename(argv$phenoFile)), ".gene_list.tsv"))
    gl <- tmpl %>% filter(.data[[key]] %in% sp$.gname) %>% rename(`#chr` = chr)
    vroom::vroom_write(gl, gl_out, delim = "\t")

  } else if (mtt == "atac") {
    tmpl <- vroom::vroom(argv$coordinate_annotation, delim = "\t",
                         col_types = cols(.default = "c"), show_col_types = FALSE)
    if ("chr" %in% names(tmpl) && !("#chr" %in% names(tmpl)))
      names(tmpl)[names(tmpl) == "chr"] <- "#chr"
    dfj <- df; names(dfj)[1] <- "ID"
    bed <- dfj %>% inner_join(tmpl, by = "ID") %>%
      add_count(ID, name = ".n") %>% filter(.data$.n == 1) %>% select(-.n)
    bed <- bed %>% select(any_of(c("#chr", "start", "end", "ID")), all_of(sample_cols))
    write_bed_r(bed, argv$output_bed)
    write_region_list(bed)
    warning("Gene partitioning is not applicable for ATAC; no gene_list.tsv generated.")
  } else {
    stop(sprintf("Unsupported molecular_trait_type: %s", mtt))
  }
  message(sprintf("Written: %s", argv$output_bed))
}

# ---------------------------------------------------------------------------
# Leafcutter cluster -> gene mapping
# ---------------------------------------------------------------------------
get_intron_meta <- function(ids) {
  parts <- strsplit(as.character(ids), ":", fixed = TRUE)
  n <- max(lengths(parts))
  mat <- do.call(rbind, lapply(parts, function(p) { length(p) <- n; p }))
  m <- as_tibble(mat, .name_repair = "minimal")
  names(m) <- if (n >= 5) c("chr", "start", "end", "clu", "category")[1:n] else
    c("chr", "start", "end", "clu")
  m %>% mutate(start = as.numeric(start), end = as.numeric(end))
}

harmonize_chr_style <- function(intron_meta, exon_tbl) {
  ih <- startsWith(as.character(intron_meta$chr[1]), "chr")
  eh <- startsWith(as.character(exon_tbl$chr[1]), "chr")
  if (!ih && eh) exon_tbl$chr <- sub("^chr", "", exon_tbl$chr)
  else if (ih && !eh) exon_tbl$chr <- paste0("chr", exon_tbl$chr)
  exon_tbl
}

map_clusters_site <- function(intron_meta, exon_tbl) {
  pieces <- list()
  for (chrom in sort(unique(stats::na.omit(intron_meta$chr)))) {
    ic <- intron_meta %>% filter(chr == chrom)
    ec <- exon_tbl %>% filter(chr == chrom)
    if (nrow(ic) == 0 || nrow(ec) == 0) next
    three <- inner_join(ic %>% transmute(temp = end, clu),
                        ec %>% transmute(temp = start, gene_name),
                        by = "temp", relationship = "many-to-many")
    five  <- inner_join(ic %>% transmute(temp = start, clu),
                        ec %>% transmute(temp = end, gene_name),
                        by = "temp", relationship = "many-to-many")
    m <- bind_rows(three, five) %>% distinct(clu, gene_name)
    if (nrow(m) == 0) next
    m$clu <- paste0(chrom, ":", m$clu)
    pieces[[length(pieces) + 1]] <- m
  }
  if (!length(pieces)) return(tibble(clu = character(), genes = character()))
  bind_rows(pieces) %>% group_by(clu) %>%
    summarise(genes = paste(gene_name, collapse = ","), .groups = "drop")
}

map_clusters_region <- function(intron_meta, exon_tbl, overlap_ratio) {
  genes <- exon_tbl %>% group_by(chr, gene_id) %>%
    summarise(start = min(start), end = max(end), .groups = "drop")
  gg <- GRanges(genes$chr, IRanges(genes$start, genes$end)); gg$gene_id <- genes$gene_id
  ii <- GRanges(intron_meta$chr, IRanges(intron_meta$start, intron_meta$end))
  ii$clu <- intron_meta$clu
  hits <- findOverlaps(ii, gg)
  if (length(hits) == 0) return(tibble(clu = character(), genes = character()))
  ov <- width(pintersect(ii[queryHits(hits)], gg[subjectHits(hits)]))
  ilen <- pmax(width(ii)[queryHits(hits)], 1)
  keep <- ov / ilen >= overlap_ratio
  tibble(clu = paste0(as.character(seqnames(ii))[queryHits(hits)][keep], ":",
                      ii$clu[queryHits(hits)][keep]),
         gene_id = gg$gene_id[subjectHits(hits)][keep]) %>%
    distinct(clu, gene_id) %>% group_by(clu) %>%
    summarise(genes = paste(gene_id, collapse = ","), .groups = "drop")
}

map_leafcutter_cluster_to_gene <- function(argv) {
  gr <- rtracklayer::import(argv$annotation_gtf)
  exon_df <- exon_table_from_gtf(gr)
  vroom::vroom_write(exon_df, argv$output_exon_list, delim = "\t")

  ic <- vroom::vroom(argv$intron_count, delim = "\t",
                     col_types = cols(.default = "c"), show_col_types = FALSE)
  intron_meta <- get_intron_meta(ic[[4]])
  exon_tbl <- harmonize_chr_style(intron_meta, exon_df)
  exon_tbl$gene_name <- exon_tbl$gene_id                  # mirror the .py overwrite
  mapping <- if (argv$map_stra == "site") map_clusters_site(intron_meta, exon_tbl)
             else if (argv$map_stra == "region")
               map_clusters_region(intron_meta, exon_tbl, argv$overlap_ratio)
             else stop("--map-stra must be 'site' or 'region'")
  vroom::vroom_write(mapping, argv$output_cluster_map, delim = "\t")
  message(sprintf("Written: %s, %s (%d clusters mapped)",
                  argv$output_exon_list, argv$output_cluster_map, nrow(mapping)))
}

# ---------------------------------------------------------------------------
# Leafcutter isoform annotation
# ---------------------------------------------------------------------------
annotate_leafcutter_isoforms <- function(argv) {
  gr <- rtracklayer::import(argv$annotation_gtf)
  tss <- tss_bed(gr, "gene", "gene_id")
  bed <- vroom::vroom(argv$phenoFile, delim = "\t",
                      col_types = cols(.default = "c"), show_col_types = FALSE)
  names(bed)[1] <- "#chr"
  id_col <- if ("ID" %in% names(bed)) "ID" else names(bed)[4]
  sample_cols <- names(bed)[5:ncol(bed)]

  cm <- vroom::vroom(argv$cluster_map, delim = "\t",
                     col_types = cols(.default = "c"), show_col_types = FALSE)
  cmap <- setNames(cm$genes, cm[[1]])

  parts <- strsplit(as.character(bed[[id_col]]), ":", fixed = TRUE)
  bed$cluster_id <- vapply(parts, function(x)
    if (length(x) >= 4) paste0(x[1], ":", x[4]) else NA_character_, character(1))
  b <- bed %>%
    filter(!is.na(cluster_id), cluster_id %in% names(cmap)) %>%
    mutate(genes = unname(cmap[cluster_id])) %>%
    separate_rows(genes, sep = ",") %>%
    filter(genes %in% tss$gene_id) %>%
    mutate(gene_isoform_id = paste0(.data[[id_col]], ":", genes)) %>%
    left_join(tss %>% select(genes = gene_id, tchr = chr, tstart = start, tend = end),
              by = "genes") %>%
    arrange(tchr, tstart)

  gene_bed <- bind_cols(
    b %>% transmute(`#chr` = tchr, start = tstart, end = tend, ID = gene_isoform_id),
    b %>% select(all_of(sample_cols)))
  # strip ".<...>" from junc-style sample columns, then apply participant map
  sc <- names(gene_bed)[5:ncol(gene_bed)]
  names(gene_bed)[5:ncol(gene_bed)] <- ifelse(grepl("junc", sc), sub("\\..*", "", sc), sc)
  gene_bed <- apply_participant_map(gene_bed, read_participant_map(argv$sample_participant_lookup))
  gene_bed <- distinct(gene_bed)
  write_bed_r(gene_bed, argv$output_bed)

  group_s <- b %>% transmute(ID = gene_isoform_id, gene = genes) %>%
    distinct() %>% arrange(ID)
  vroom::vroom_write(group_s, argv$output_phenotype_group, delim = "\t")
  message(sprintf("Written: %s, %s", argv$output_bed, argv$output_phenotype_group))
}

# ---------------------------------------------------------------------------
# Psichomics isoform annotation
# ---------------------------------------------------------------------------
annotate_psichomics_isoforms <- function(argv) {
  gr <- rtracklayer::import(argv$annotation_gtf)
  tss <- tss_bed(gr, "gene", "gene_id")
  bed <- vroom::vroom(argv$phenoFile, delim = "\t",
                      col_types = cols(.default = "c"), show_col_types = FALSE)
  last_field <- function(s, i_from_end)
    vapply(strsplit(as.character(s), "_", fixed = TRUE),
           function(x) x[length(x) - i_from_end], character(1))
  bed$gene_id <- last_field(bed$ID, 0)
  bed <- bed %>% select(-any_of(c("#Chr", "#chr", "start", "end")))

  out <- tss %>% right_join(bed, by = "gene_id") %>% arrange(chr, start)
  out <- apply_participant_map(out, read_participant_map(argv$sample_participant_lookup))
  out <- out %>%
    mutate(gene_type_id = vapply(strsplit(as.character(ID), "_", fixed = TRUE),
                                 function(x) paste0(x[length(x)], "_", x[1]), character(1)))
  phenotype_group <- out %>% select(ID, gene_type_id)
  bed_out <- out %>% select(-gene_id, -gene_type_id) %>%
    select(`#chr` = chr, start, end, ID, everything())
  write_bed_r(bed_out, argv$output_bed)
  vroom::vroom_write(phenotype_group, argv$output_phenotype_group, delim = "\t", col_names = FALSE)
  message(sprintf("Written: %s, %s", argv$output_bed, argv$output_phenotype_group))
}

# ---------------------------------------------------------------------------
# annotate_coord_biomart (Ensembl web service; needs network)
# ---------------------------------------------------------------------------
annotate_coord_biomart <- function(argv) {
  suppressPackageStartupMessages(library(biomaRt))
  if (!have(argv$phenoFile) || !have(argv$output_bed) || !have(argv$output_region_list))
    stop("annotate_coord_biomart requires --phenoFile, --output-bed and --output-region-list")
  if (is.na(argv$ensembl_version))
    stop("--ensembl-version is required for annotate_coord_biomart")

  dir.create(dirname(argv$output_bed), recursive = TRUE, showWarnings = FALSE)
  biomartCacheClear()
  gene_exp <- vroom::vroom(argv$phenoFile, delim = "\t", show_col_types = FALSE)
  if ("#chr" %in% colnames(gene_exp)) gene_exp <- gene_exp[, 4:ncol(gene_exp)]
  id_col <- if ("gene_ID" %in% colnames(gene_exp)) "gene_ID"
            else if ("gene_id" %in% colnames(gene_exp)) "gene_id"
            else colnames(gene_exp)[1]

  ensembl <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl",
                        version = argv$ensembl_version)
  ensembl_df <- getBM(attributes = c("ensembl_gene_id", "chromosome_name",
                                     "start_position", "end_position"), mart = ensembl)
  # biomaRt pulls in AnnotationDbi::select (masks dplyr); qualify dplyr verbs here.
  my_genes_ann <- ensembl_df %>%
    dplyr::filter(.data$ensembl_gene_id %in% gene_exp[[id_col]],
                  .data$chromosome_name %in% as.character(1:23)) %>%
    dplyr::rename(`#chr` = "chromosome_name", start = "start_position",
                  end = "end_position", gene_ID = "ensembl_gene_id") %>%
    dplyr::filter(.data$gene_ID != "NA")
  my_genes_ann %>% dplyr::select(`#chr`, start, end, gene_ID) %>%
    vroom::vroom_write(argv$output_region_list, delim = "\t")

  colnames(gene_exp)[colnames(gene_exp) == id_col] <- "gene_ID"
  my_gene_bed <- my_genes_ann %>% dplyr::mutate(end = .data$start + 1) %>%
    dplyr::inner_join(gene_exp, by = "gene_ID") %>% dplyr::arrange(`#chr`, start) %>%
    dplyr::select(`#chr`, start, end, gene_ID, dplyr::everything())  # #chr first for a valid BED
  plain <- sub("\\.gz$", "", argv$output_bed)
  vroom::vroom_write(my_gene_bed, plain, delim = "\t")
  Rsamtools::bgzip(plain, dest = argv$output_bed, overwrite = TRUE); file.remove(plain)
  Rsamtools::indexTabix(argv$output_bed, format = "bed")
  message(sprintf("Written: %s", argv$output_bed))
}

# ---------------------------------------------------------------------------
switch(argv$step,
  annotate_coord                 = annotate_coord(argv),
  map_leafcutter_cluster_to_gene = map_leafcutter_cluster_to_gene(argv),
  annotate_leafcutter_isoforms   = annotate_leafcutter_isoforms(argv),
  annotate_psichomics_isoforms   = annotate_psichomics_isoforms(argv),
  annotate_coord_biomart         = annotate_coord_biomart(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
