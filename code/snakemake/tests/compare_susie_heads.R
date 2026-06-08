args <- commandArgs(trailingOnly = TRUE)
old_bvsr <- args[[1]]
modular_sos_bvsr <- args[[2]]
old_twas <- args[[3]]
modular_sos_twas <- args[[4]]
compare_tsv <- args[[5]]
report_txt <- args[[6]]

extract_sumstats <- function(path) {
  obj <- readRDS(path)
  x <- obj[[1]][[1]]$sumstats
  data.frame(
    variant = names(x$betahat),
    betahat = unname(x$betahat),
    sebetahat = unname(x$sebetahat),
    z_scores = unname(x$z_scores),
    check.names = FALSE
  )
}

extract_twas <- function(path) {
  obj <- readRDS(path)
  x <- obj[[1]][[1]]$twas_weights
  data.frame(
    variant = rownames(x$enet_weights),
    enet = unname(x$enet_weights[, 1]),
    lasso = unname(x$lasso_weights[, 1]),
    bayes_r = unname(x$bayes_r_weights[, 1]),
    bayes_l = unname(x$bayes_l_weights[, 1]),
    mrash = unname(x$mrash_weights[, 1]),
    susie = unname(x$susie_weights[, 1]),
    check.names = FALSE
  )
}

old_sum <- utils::head(extract_sumstats(old_bvsr), 3)
modular_sos_sum <- utils::head(extract_sumstats(modular_sos_bvsr), 3)
old_twas_df <- utils::head(extract_twas(old_twas), 3)
modular_sos_twas_df <- utils::head(extract_twas(modular_sos_twas), 3)

susie_match <- identical(old_sum, modular_sos_sum)
twas_match <- identical(old_twas_df, modular_sos_twas_df)

write.table(
  data.frame(
    metric = c("susie_head", "twas_head"),
    old = c("present", "present"),
    modular_sos = c("present", "present"),
    status = c(if (susie_match) "match" else "diff", if (twas_match) "match" else "diff")
  ),
  file = compare_tsv,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

sink(report_txt)
cat("20_susie_head_match:", if (susie_match) "TRUE" else "FALSE", "\n")
cat("20_twas_head_match:", if (twas_match) "TRUE" else "FALSE", "\n\n")
cat("20_susie_head_old:\n")
print(old_sum)
cat("\n20_susie_head_modular_sos:\n")
print(modular_sos_sum)
cat("\n20_twas_head_old:\n")
print(old_twas_df)
cat("\n20_twas_head_modular_sos:\n")
print(modular_sos_twas_df)
sink()

if (!susie_match || !twas_match) {
  quit(status = 1)
}
