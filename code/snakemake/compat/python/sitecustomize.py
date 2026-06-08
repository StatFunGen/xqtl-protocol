import os
import csv
import shutil
import subprocess


def _local_rscript():
    helper_bin = os.environ.get("XQTL_LOCAL_PIXI_HELPER_BIN")
    if helper_bin:
        candidate = os.path.join(helper_bin, "Rscript")
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("Rscript")


def _patch_rpy2_rscript_probe():
    rscript = _local_rscript()
    if not rscript:
        return

    try:
        import rpy2.situation as situation
    except Exception:
        return

    def r_ld_library_path_from_subprocess(r_home):
        cmd = (rscript, "-e", 'cat(Sys.getenv("LD_LIBRARY_PATH"))')
        try:
            return subprocess.check_output(
                cmd,
                universal_newlines=True,
                stderr=subprocess.PIPE,
            )
        except Exception:
            return ""

    situation.r_ld_library_path_from_subprocess = r_ld_library_path_from_subprocess

    try:
        import rpy2.rinterface as rinterface
    except Exception:
        return

    def _getrenvvars(baselinevars=None, r_home=None):
        if baselinevars is None:
            baselinevars = os.environ
        cmd = (
            rscript,
            "-e",
            "x<-Sys.getenv();y<-as.character(x);names(y)<-names(x);write.csv(y)",
        )
        envvars = subprocess.check_output(
            cmd,
            universal_newlines=True,
            stderr=subprocess.PIPE,
        )
        reader = csv.reader(row for row in envvars.split(os.linesep) if row)
        next(reader)
        return tuple((k, v) for k, v in reader if k not in baselinevars)

    rinterface._getrenvvars = _getrenvvars


def _patch_tensorqtl_sort_check():
    if os.environ.get("XQTL_PATCH_TENSORQTL_SORT") != "1":
        return

    import pandas as pd
    import tensorqtl
    import tensorqtl.core as tensorqtl_core

    def read_phenotype_bed_compat(phenotype_bed):
        """Match legacy tensorqtl I/O while tolerating current pandas groupby behavior."""
        if phenotype_bed.lower().endswith((".bed.gz", ".bed")):
            phenotype_df = pd.read_csv(
                phenotype_bed,
                sep="\t",
                index_col=3,
                dtype={"#chr": str, "#Chr": str},
            )
        elif phenotype_bed.lower().endswith(".bed.parquet"):
            phenotype_df = pd.read_parquet(phenotype_bed)
            phenotype_df.set_index(phenotype_df.columns[3], inplace=True)
        else:
            raise ValueError("Unsupported file type.")

        phenotype_df.rename(
            columns={col: col.lower().replace("#chr", "chr") for col in phenotype_df.columns[:3]},
            inplace=True,
        )
        phenotype_df["start"] += 1
        pos_df = phenotype_df[["chr", "start", "end"]].copy()
        phenotype_df.drop(["chr", "start", "end"], axis=1, inplace=True)

        sorted_pos_df = pos_df.sort_values(["chr", "start", "end"], kind="mergesort")
        assert pos_df.reset_index(drop=True).equals(
            sorted_pos_df.reset_index(drop=True)
        ), "Positions in BED file must be sorted."

        if (pos_df["start"] == pos_df["end"]).all():
            pos_df = pos_df[["chr", "end"]].rename(columns={"end": "pos"})

        return phenotype_df, pos_df

    tensorqtl_core.read_phenotype_bed = read_phenotype_bed_compat
    tensorqtl.read_phenotype_bed = read_phenotype_bed_compat

    try:
        import tensorqtl.tensorqtl as tensorqtl_cli
    except Exception:
        return
    tensorqtl_cli.read_phenotype_bed = read_phenotype_bed_compat


try:
    _patch_rpy2_rscript_probe()
except Exception:
    # Keep startup non-fatal for unrelated Python entrypoints.
    pass

try:
    _patch_tensorqtl_sort_check()
except Exception:
    # Keep startup non-fatal for unrelated Python entrypoints.
    pass
