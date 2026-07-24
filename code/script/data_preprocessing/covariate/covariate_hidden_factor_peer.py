#!/usr/bin/env python3
# ============================================================
# covariate_hidden_factor_peer.py
# Mirrors the PEER step of:
#   code/data_preprocessing/covariate/covariate_hidden_factor.ipynb
#
# Fits a MOFA model to a residualized molecular-phenotype matrix using the
# mofapy2 Python package (the GTEx-style "PEER" replacement), saves the HDF5
# model exactly as before, and ALSO extracts everything the R diagnostics
# need (factors, weights, per-factor + total variance explained) into plain
# TSV files that R can read WITHOUT the MOFA2 Bioconductor package / reticulate.
#
# This is the whole Python surface of the notebook: mofapy2 stays external,
# but the R worker calls this as a proper standalone script (not a generated
# heredoc) and reads only the TSV sidecars afterwards.
# ============================================================
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd


def gtex_num_factors(n_samples):
    """GTeX recommendation for the number of hidden factors by sample size."""
    if n_samples < 150:
        return 15
    elif n_samples < 250:
        return 30
    elif n_samples < 350:
        return 45
    return 60


def parse_args():
    p = argparse.ArgumentParser(description="Fit mofapy2 PEER model + extract factors/weights/variance")
    p.add_argument("--resid-file", required=True, help="Residual phenotype .bed.gz from compute_residual")
    p.add_argument("--model-file", required=True, help="Output HDF5 model path (.PEER_MODEL.hd5)")
    p.add_argument("--factors-out", required=True, help="Output factors TSV (factor x sample)")
    p.add_argument("--weights-out", required=True, help="Output weights TSV (feature x factor)")
    p.add_argument("--variance-out", required=True, help="Output variance-explained TSV (factor, r2)")
    p.add_argument("--num-factor", type=int, default=0, help="Number of factors (0 = GTeX default by sample size)")
    p.add_argument("--iteration", type=int, default=1000, help="Max training iterations")
    p.add_argument("--convergence-mode", default="fast", help="MOFA convergence mode: fast/medium/slow")
    p.add_argument("--num-threads", default="8", help="BLAS thread count")
    p.add_argument("--tol", type=float, default=0.001, help="Convergence tolerance")
    p.add_argument("--r2-tol", default="False", help="Optional dropR2 setting; False disables it")
    return p.parse_args()


def first_child(h5grp):
    """The single group/view child key (MOFA writes exactly one here)."""
    return list(h5grp.keys())[0]


def main():
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = args.num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = args.num_threads
    os.environ["MKL_NUM_THREADS"] = args.num_threads

    # ── Load residual matrix (genes x samples) ────────────────────────────────
    data = pd.read_csv(args.resid_file, sep="\t", index_col=3).drop(["#chr", "start", "end"], axis=1)

    num_factor = args.num_factor
    if num_factor == 0:
        num_factor = gtex_num_factors(len(data.columns))

    # ── Fit MOFA (mofapy2) ────────────────────────────────────────────────────
    from mofapy2.run.entry_point import entry_point

    ent = entry_point()
    ent.set_data_matrix(
        [[data.transpose()]],
        samples_names=[data.columns.values.tolist()],
        features_names=[data.index.values.tolist()],
    )
    ent.set_model_options(factors=num_factor, spikeslab_weights=False, ard_weights=False)
    train_options = dict(
        iter=args.iteration,
        convergence_mode=args.convergence_mode,
        startELBO=1,
        freqELBO=1,
        tolerance=args.tol,
        gpu_mode=False,
        verbose=True,
        seed=42,
    )
    r2_tol = str(args.r2_tol).lower()
    if r2_tol not in ("false", "f", "0", "none", "null", ""):
        if r2_tol in ("true", "t", "yes", "y"):
            train_options["dropR2"] = True
        else:
            train_options["dropR2"] = float(args.r2_tol)
    ent.set_train_options(**train_options)
    ent.build()
    ent.run()
    ent.save(args.model_file)

    # Re-encode feature names as bytes (matches the previous notebook behaviour)
    right_name = [x.encode("UTF-8") for x in ent.data_opts["features_names"][0]]
    with h5py.File(args.model_file, "r+") as new_hd5:
        del new_hd5["features/view0"]
        new_hd5["features"].create_dataset("view0", data=np.array(right_name))

    # ── Extract factors / weights / variance for R ────────────────────────────
    with h5py.File(args.model_file, "r") as f:
        grp = first_child(f["expectations/Z"])
        view = first_child(f["expectations/W"])
        Z = f["expectations/Z/" + grp][:]          # (K, N)
        W = f["expectations/W/" + view][:]          # (K, D)
        Y = f["data/" + view + "/" + grp][:]        # (N, D)
        intercept = f["intercepts/" + view + "/" + grp][:]  # (D,)
        r2_total = float(np.ravel(f["variance_explained/r2_total/" + grp][:])[0])
        samples = [s.decode() if isinstance(s, bytes) else str(s) for s in f["samples/" + grp][:]]
        features = [s.decode() if isinstance(s, bytes) else str(s) for s in f["features/" + view][:]]

    K = Z.shape[0]
    factor_names = ["Factor%d" % (k + 1) for k in range(K)]

    # Per-factor variance explained (MOFA definition): percentage of centred-data
    # variance reconstructed by each single factor. MOFA reports r2 in percent, so
    # r2_total (read from the model) and these per-factor values share that scale.
    Yc = Y - intercept[np.newaxis, :]
    ss_tot = float(np.sum(Yc ** 2))
    Zt = Z.T                                        # (N, K)
    r2_per_factor = []
    for k in range(K):
        pred_k = np.outer(Zt[:, k], W[k, :])        # (N, D)
        r2_k = 1.0 - float(np.sum((Yc - pred_k) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        r2_per_factor.append(100.0 * r2_k)

    # factors.tsv: factor x sample, first column '#id'
    factors_df = pd.DataFrame(Zt.T, index=factor_names, columns=samples)  # (K, N)
    factors_df.insert(0, "#id", factor_names)
    factors_df.to_csv(args.factors_out, sep="\t", index=False)

    # weights.tsv: feature x factor, first column 'feature'
    weights_df = pd.DataFrame(W.T, index=features, columns=factor_names)  # (D, K)
    weights_df.insert(0, "feature", features)
    weights_df.to_csv(args.weights_out, sep="\t", index=False)

    # variance.tsv: factor, r2 (per factor) + a 'Total' row
    variance_df = pd.DataFrame(
        {"factor": factor_names + ["Total"], "r2": r2_per_factor + [r2_total]}
    )
    variance_df.to_csv(args.variance_out, sep="\t", index=False)

    sys.stderr.write(
        "PEER (mofapy2) fit: %d factors x %d samples, %d features; model=%s\n"
        % (K, len(samples), len(features), args.model_file)
    )


if __name__ == "__main__":
    main()
