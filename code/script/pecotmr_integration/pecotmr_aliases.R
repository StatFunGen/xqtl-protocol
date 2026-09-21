# pecotmr_aliases.R
#
# Shared name aliases for the pecotmr_integration wrapper scripts. Each script
# sources this from its own directory:
#
#   .d <- dirname(sub("^--file=", "",
#           grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
#   source(file.path(.d, "pecotmr_aliases.R"))
#
# pecotmr 0.8.2 camelCased its MASH component/Vhat names and its TWAS method
# tokens. The snake_case spellings remain the public interface on this side --
# they are SoS step names in mixture_prior.ipynb (`[flash_nonneg]`,
# `[vhat_simple_specific]`), they appear in output filenames that downstream
# steps consume, and they are the documented `--twas-methods` values in
# mnm_regression.ipynb -- so the wrappers keep accepting them and translate
# here. Both spellings are accepted, so callers already using the pecotmr
# names work unchanged.
#
# This is argument marshalling only; no analysis logic (that lives in pecotmr).

# mashCovarianceComponents(components=) / mashPriorCovariances(components=)
MASH_COMPONENT_ALIASES <- c(flash_nonneg = "flashNonneg")

# mashResidualCorrelation(method=)
MASH_VHAT_ALIASES <- c(simple_specific = "simpleSpecific")

# twasWeightsPipeline(methods=). Mirrors pecotmr's
# .twasKnownMethodLookupNames(); tokens that are already single words
# (susie, mrash, enet, lasso, scad, mcp, l0learn, mvsusie, mrmash) need no alias.
TWAS_METHOD_ALIASES <- c(
  susie_ash          = "susieAsh",
  susie_inf          = "susieInf",
  bayes_r            = "bayesR",
  bayes_l            = "bayesL",
  bayes_a            = "bayesA",
  bayes_b            = "bayesB",
  bayes_c            = "bayesC",
  bayes_n            = "bayesN",
  b_lasso            = "bLasso",
  dpr_vb             = "dprVb",
  dpr_gibbs          = "dprGibbs",
  dpr_adaptive_gibbs = "dprAdaptiveGibbs"
)

# Map any aliased names to the spelling pecotmr expects, passing others through
# unchanged so pecotmr reports unknown values itself.
apply_pecotmr_aliases <- function(values, aliases) {
  hit <- match(values, names(aliases))
  ifelse(is.na(hit), values, unname(aliases[hit]))
}

# Split a comma/space separated CLI list and normalise the names in it.
split_pecotmr_names <- function(value, aliases, split = "[ ,]+") {
  parts <- trimws(strsplit(value, split)[[1L]])
  parts <- parts[nzchar(parts)]
  apply_pecotmr_aliases(parts, aliases)
}
