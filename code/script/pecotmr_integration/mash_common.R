# mash_common.R
#
# Shared helpers for the pecotmr_integration MASH wrapper scripts
# (mash_covariance / mash_vhat / mash_prior). Each script sources this from its
# own directory:
#
#   .d <- dirname(sub("^--file=", "",
#           grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
#   source(file.path(.d, "mash_common.R"))
#
# pecotmr 0.8.2 camelCased two MASH names: the `flash_nonneg` covariance
# component became `flashNonneg`, and the `simple_specific` Vhat method became
# `simpleSpecific`. The snake_case spellings are still the public interface on
# this side -- they are SoS step names in mixture_prior.ipynb (`[flash_nonneg]`,
# `[vhat_simple_specific]`) and they appear in the output filenames downstream
# steps consume -- so the wrappers keep accepting them and translate here.
# Both spellings are accepted, so callers already using the pecotmr names work.
#
# This is argument marshalling only; no analysis logic (that lives in pecotmr).

MASH_COMPONENT_ALIASES <- c(flash_nonneg = "flashNonneg")
MASH_VHAT_ALIASES <- c(simple_specific = "simpleSpecific")

# Map any aliased names to the spelling pecotmr expects, passing others through
# unchanged so pecotmr reports unknown values itself.
apply_mash_aliases <- function(values, aliases) {
  hit <- match(values, names(aliases))
  ifelse(is.na(hit), values, unname(aliases[hit]))
}

# Split a comma/space separated CLI list and normalise the names in it.
split_mash_names <- function(value, aliases) {
  parts <- trimws(strsplit(value, "[ ,]+")[[1L]])
  parts <- parts[nzchar(parts)]
  apply_mash_aliases(parts, aliases)
}
