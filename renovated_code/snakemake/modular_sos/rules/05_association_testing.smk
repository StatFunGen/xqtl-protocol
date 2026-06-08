# ============================================================
# Rule Module 05: QTL Association Testing  (Modular SoS)
# ============================================================
# Covers: CIS-QTL association testing with TensorQTL
#
# SoS notebook called (Modular SoS wrapper in renovated_code/notebook/modular_sos/):
#   - TensorQTL.ipynb (cis)
#
# The notebook task block calls:
#   renovated_code/association_scan/TensorQTL/TensorQTL.sh cis
# ============================================================

# ------------------------------------
# Step 5.1 — TensorQTL CIS association scan
# ------------------------------------
rule tensorqtl_cis:
    """Run TensorQTL cis-QTL nominal + permutation scan across all chromosomes."""
    input:
        geno_list      = lambda wc: get_genotype_chrom_list(),
        phenotype_bed  = lambda wc: get_phenotype_bed(wc),
        pheno_list     = lambda wc: get_phenotype_chrom_list(wc.theme),
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        done = "{cwd}/association_scan/{theme}/TensorQTL/.done_cis",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("tensorqtl_cis"),
        tensorqtl_notebook = f"{NOTEBOOKS}/TensorQTL.ipynb",
        compat_python  = COMPAT_PYTHON,
        outdir       = "{cwd}/association_scan/{theme}/TensorQTL",
        normalized_geno_manifest = lambda wc: (
            f"{config['cwd']}/association_scan/{wc.theme}/TensorQTL/"
            f"{PLINK_QC_BASENAME}.genotype_by_chrom_files.normalized.txt"
        ),
        normalized_pheno_manifest = lambda wc: (
            f"{config['cwd']}/association_scan/{wc.theme}/TensorQTL/"
            f"{get_phenotype_base(wc.theme)}.phenotype_by_chrom_files.normalized.txt"
        ),
        cis_window   = config["association"]["cis_window"],
        mac          = config["association"]["mac_threshold"],
        maf          = config["association"]["maf_threshold"],
        sos_mem      = f"{config['resources']['tensorqtl']['mem_mb'] // 1000}G",
        sos_walltime = sos_walltime_arg(config["resources"]["tensorqtl"]["runtime"], sos_queue("tensorqtl_cis")),
        dry_run      = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["tensorqtl"]["mem_mb"],
        runtime  = config["resources"]["tensorqtl"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        awk 'BEGIN{{FS=OFS="\\t"}} NR==1 {{print "#id","#path"; next}} {{$1=$1; sub(/^chr/, "", $1); print $1, $2}}' {input.geno_list} > {params.normalized_geno_manifest}
        awk 'BEGIN{{FS=OFS="\\t"}} NR==1 {{print "#id","#path"; next}} {{$1=$1; sub(/^chr/, "", $1); print $1, $2}}' {input.pheno_list} > {params.normalized_pheno_manifest}
        geno_rows="$(awk 'END {{print NR-1}}' {params.normalized_geno_manifest})"
        pheno_rows="$(awk 'END {{print NR-1}}' {params.normalized_pheno_manifest})"
        chrom_args="$(awk 'NR>1 {{print $1}}' {params.normalized_geno_manifest} | paste -sd ' ' -)"
        if [ -z "$chrom_args" ]; then
            echo "ERROR: normalized genotype manifest is empty" >&2
            exit 1
        fi
	        if [ "$geno_rows" -eq 1 ] && [ "$pheno_rows" -eq 1 ]; then
	            geno_bed="$(awk 'NR==2 {{print $2}}' {params.normalized_geno_manifest})"
	            XQTL_PATCH_TENSORQTL_SORT=1 PYTHONPATH="{params.compat_python}${{PYTHONPATH:+:$PYTHONPATH}}" \
	            {params.sos_bin} run {params.tensorqtl_notebook} cis {params.dry_run} \
	                --cwd {params.outdir} \
	                --genotype-file "$geno_bed" \
	                --phenotype-file {input.phenotype_bed} \
	                --chromosome $chrom_args \
	                --covariate-file {input.hidden_factors} \
		                --window {params.cis_window} \
		                --MAC {params.mac} \
		                --maf-threshold {params.maf} \
		                --walltime {params.sos_walltime} \
		                --mem {params.sos_mem} \
		                --numThreads {threads} {params.sos_sched}
	        else
	            XQTL_PATCH_TENSORQTL_SORT=1 PYTHONPATH="{params.compat_python}${{PYTHONPATH:+:$PYTHONPATH}}" \
	            {params.sos_bin} run {params.tensorqtl_notebook} cis {params.dry_run} \
	                --cwd {params.outdir} \
	                --genotype-file {params.normalized_geno_manifest} \
	                --phenotype-file {params.normalized_pheno_manifest} \
	                --chromosome $chrom_args \
	                --covariate-file {input.hidden_factors} \
		                --window {params.cis_window} \
		                --MAC {params.mac} \
		                --maf-threshold {params.maf} \
		                --walltime {params.sos_walltime} \
		                --mem {params.sos_mem} \
		                --numThreads {threads} {params.sos_sched}
	        fi
	        python3 - "{params.outdir}" "{params.normalized_pheno_manifest}" <<'PY'
import csv
import os
import sys

outdir, manifest = sys.argv[1], sys.argv[2]

def strip_exts(name, count):
    for _ in range(count):
        name = os.path.splitext(name)[0]
    return name

rows = []
with open(manifest, newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        chrom = str(row.get("#id") or row.get("id") or "").replace("chr", "")
        path = row.get("#path") or row.get("path")
        if path:
            rows.append((chrom, path))

if not rows:
    print("ERROR: TensorQTL phenotype manifest has no rows", file=sys.stderr)
    sys.exit(1)

missing = []
nominal_names = []
for chrom, path in rows:
    stem = strip_exts(os.path.basename(path), 2)
    chrom_suffix = "" if chrom == "0" else "_chr%s" % chrom
    parquet_chrom = "" if chrom == "0" else chrom
    nominal = "%s%s.cis_qtl.pairs.tsv.gz" % (stem, chrom_suffix)
    nominal_names.append(nominal)
    expected = [
        "%s.cis_qtl_pairs.%s.parquet" % (stem, parquet_chrom),
        nominal,
        nominal + ".tbi",
        "%s%s.cis_qtl.regional.tsv.gz" % (stem, chrom_suffix),
        "%s%s.cis_qtl.regional.tsv.gz.tbi" % (stem, chrom_suffix),
    ]
    missing.extend(name for name in expected if not os.path.isfile(os.path.join(outdir, name)))

if len(rows) > 1:
    sig_prefix = strip_exts(os.path.basename(rows[0][1]), 3)
else:
    sig_prefix = strip_exts(nominal_names[0], 4)

for name in [
    sig_prefix + ".cis_qtl_regional_significance.tsv.gz",
    sig_prefix + ".cis_qtl_regional_significance.summary.txt",
]:
    if not os.path.isfile(os.path.join(outdir, name)):
        missing.append(name)

if missing:
    print("ERROR: TensorQTL missing expected output(s):", file=sys.stderr)
    for name in missing[:20]:
        print("  " + name, file=sys.stderr)
    if len(missing) > 20:
        print("  ... %d more" % (len(missing) - 20), file=sys.stderr)
    sys.exit(1)
PY
	        tensorqtl_guard_status=$?
	        if [ "$tensorqtl_guard_status" -ne 0 ]; then
	            exit "$tensorqtl_guard_status"
	        fi
	        touch {output.done}
	        """
