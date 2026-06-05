#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MWE_DIR="${MWE_DIR:-}"
if [[ -z "${MWE_DIR}" ]]; then
    MWE_DIR="$(cd -- "${ROOT}/../mwe_data" && pwd)"
else
    MWE_DIR="$(cd -- "${MWE_DIR}" && pwd)"
fi
T="${1:-${ROOT}/tmp/modular_sos_mwe_test}"

if [[ "${XQTL_MODULAR_SOS_SKIP_PIXI:-0}" != "1" ]]; then
    source "${ROOT}/renovated_code/snakemake/dryrun/activate_local_pixi.sh" >/dev/null
fi

mkdir -p "${T}" "${T}/output"

link_into_test() {
    local src="$1"
    local dst="$2"
    ln -sfn "${src}" "${dst}"
}

for prefix in AC.unrelated AC.related; do
    for ext in bed bim fam; do
        link_into_test "${MWE_DIR}/${prefix}.${ext}" "${T}/${prefix}.${ext}"
    done
done

for prefix in AC.unrelated.plink_qc.prune AC.related.plink_qc.extracted; do
    for ext in bed bim fam; do
        link_into_test "${MWE_DIR}/${prefix}.${ext}" "${T}/${prefix}.${ext}"
    done
done
link_into_test "${MWE_DIR}/AC.unrelated.plink_qc.prune.in" "${T}/AC.unrelated.plink_qc.prune.in"

link_into_test "${MWE_DIR}/AC_bam_list.txt" "${T}/AC_bam_list.txt"
mkdir -p "${T}/fastq"
find "${MWE_DIR}/fastq" -maxdepth 1 -type f -exec ln -sfn {} "${T}/fastq/" \;
ln -sfn "${MWE_DIR}/fastq/sample2_r1.fastq.gz" "${T}/fastq/sample3_r1.fastq.gz"
ln -sfn "${MWE_DIR}/fastq/sample2_r2.fastq.gz" "${T}/fastq/sample3_r2.fastq.gz"
mkdir -p "${T}/bam"
find "${MWE_DIR}/bam" -maxdepth 1 -type f -exec ln -sfn {} "${T}/bam/" \;
ln -sfn "${MWE_DIR}/bam/RISK_69.Aligned.sortedByCoord.out_wasp_qc.md.bam" \
    "${T}/bam/sample3.Aligned.sortedByCoord.out_wasp_qc.md.bam"
ln -sfn "${MWE_DIR}/bam/RISK_69.Aligned.sortedByCoord.out_wasp_qc.md.bam.bai" \
    "${T}/bam/sample3.Aligned.sortedByCoord.out_wasp_qc.md.bam.bai"
ln -sfn "${MWE_DIR}/bam/RISK_69.SJ.out.tab" "${T}/bam/sample3.SJ.out.tab"
ln -sfn "${MWE_DIR}/bam/RISK_69.Aligned.toTranscriptome.out_wasp_qc.bam" \
    "${T}/bam/sample3.Aligned.toTranscriptome.out_wasp_qc.bam"

python - <<'PY' "${MWE_DIR}/AC_sample_fastq.list" "${T}/AC_sample_fastq.list"
import csv
import sys

src, dst = sys.argv[1:]
keep = ["ID", "fq1", "fq2", "strand", "read_length"]

with open(src) as inp, open(dst, "w", newline="") as out:
    reader = csv.DictReader(inp, delimiter="\t")
    fieldnames = [name for name in keep if name in reader.fieldnames]
    writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in reader:
        writer.writerow({name: row[name] for name in fieldnames})
    sample3 = {
        "ID": "sample3",
        "fq1": "sample3_r1.fastq.gz",
        "fq2": "sample3_r2.fastq.gz",
        "strand": "unstranded",
        "read_length": "100",
    }
    writer.writerow({name: sample3[name] for name in fieldnames})
PY

python - <<'PY' "${T}/AC_bam_list.txt" "${T}/AC_bam_list.normalized.txt" "${T}"
import csv
import sys
from pathlib import Path

src, dst, root = map(Path, sys.argv[1:])
root = root.resolve()
with src.open() as inp, dst.open("w", newline="") as out:
    reader = csv.DictReader(inp, delimiter="\t")
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    rows = []
    for row in reader:
        # The synthetic MWE GTF is strand-agnostic; using the original stranded
        # labels can make one tiny sample all-zero and produce RNA-SeQC NaN TPM.
        if "strand" in row:
            row["strand"] = "unstranded"
        for key in ("coord_bam_list", "SJ_list", "trans_bam_list"):
            value = row.get(key, "")
            if value.startswith("../bam/"):
                row[key] = str(root / "bam" / value.split("/", 2)[-1])
        rows.append(row)
        writer.writerow(row)
    if rows:
        sample3 = dict(rows[-1])
        sample3["sample_id"] = "sample3"
        if "coord_bam_list" in sample3:
            sample3["coord_bam_list"] = str(root / "bam" / "sample3.Aligned.sortedByCoord.out_wasp_qc.md.bam")
        if "SJ_list" in sample3:
            sample3["SJ_list"] = str(root / "bam" / "sample3.SJ.out.tab")
        if "trans_bam_list" in sample3:
            sample3["trans_bam_list"] = str(root / "bam" / "sample3.Aligned.toTranscriptome.out_wasp_qc.bam")
        writer.writerow(sample3)
PY

for asset in \
    AC.rnaseqc.gene_tpm.gct.gz \
    AC.rnaseqc.gene_reads.gct.gz \
    AC.low_expr.tpm.gct.gz \
    AC.low_expr.count.gct.gz \
    collapsed.gtf \
    ref.fasta \
    ref.fasta.fai
do
    link_into_test "${MWE_DIR}/${asset}" "${T}/${asset}"
done

cat > "${T}/rnaseqc_compatible.gtf" <<'EOF'
chr1	MWE	gene	10000	20000	.	+	.	gene_id "MWE_GENE1"; gene_name "MWE_GENE1"; transcript_id "MWE_TX1";
chr1	MWE	exon	10000	11000	.	+	.	gene_id "MWE_GENE1"; gene_name "MWE_GENE1"; transcript_id "MWE_TX1";
chr1	MWE	exon	15000	16000	.	+	.	gene_id "MWE_GENE1"; gene_name "MWE_GENE1"; transcript_id "MWE_TX1";
chr2	MWE	gene	20000	32000	.	-	.	gene_id "MWE_GENE2"; gene_name "MWE_GENE2"; transcript_id "MWE_TX2";
chr2	MWE	exon	20000	21000	.	-	.	gene_id "MWE_GENE2"; gene_name "MWE_GENE2"; transcript_id "MWE_TX2";
chr2	MWE	exon	30000	32000	.	-	.	gene_id "MWE_GENE2"; gene_name "MWE_GENE2"; transcript_id "MWE_TX2";
EOF

link_into_test "${MWE_DIR}/ZOD14598_AD_GRM_WGS_2021-04-29_all.recalibrated_variants.leftnorm.filtered.AF.WASP.vcf" \
    "${T}/genotype.full.vcf"

python - <<'PY' "${T}/genotype.full.vcf" "${T}/genotype.vcf"
import sys
from collections import defaultdict

src, dst = sys.argv[1], sys.argv[2]
targets = ["chr1", "chr2", "chr22", "1", "2", "22"]
counts = defaultdict(int)
kept = 0
drop_info_tags = {"VD"}

def strip_conflicting_info(info_field):
    if info_field in {"", "."}:
        return info_field
    kept_parts = [
        part for part in info_field.split(";")
        if part.split("=", 1)[0] not in drop_info_tags
    ]
    return ";".join(kept_parts) if kept_parts else "."

def synthetic_sample(sample_field, fmt_field, line_index, mode):
    values = sample_field.split(":")
    formats = fmt_field.split(":")
    if "GT" not in formats:
        return sample_field
    gt_idx = formats.index("GT")
    gt = values[gt_idx]
    sep = "/" if "/" in gt else "|" if "|" in gt else None
    if sep is None:
        return sample_field
    alleles = gt.replace("|", "/").split("/")
    if any(a not in {"0", "1", "."} for a in alleles):
        return sample_field
    if mode == 2 and alleles in (["0", "0"], ["0", "1"], ["1", "0"]):
        new_gt = "1/1" if line_index % 2 == 0 else "0/1"
    elif mode == 2 and alleles == ["1", "1"]:
        new_gt = "0/0"
    elif alleles == ["0", "0"]:
        new_gt = "1/1"
    elif alleles == ["1", "1"]:
        new_gt = "0/0"
    elif alleles in (["0", "1"], ["1", "0"]):
        new_gt = "0/0" if line_index % 2 == 0 else "1/1"
    else:
        new_gt = gt.replace("|", "/")
    values[gt_idx] = new_gt if sep == "/" else new_gt.replace("/", "|")
    return ":".join(values)

with open(src) as inp, open(dst, "w") as out:
    line_index = 0
    for line in inp:
        if line.startswith("##"):
            if line.startswith("##INFO=<ID=VD,"):
                continue
            out.write(line)
            continue
        if line.startswith("#CHROM"):
            fields = line.rstrip("\n").split("\t")
            samples = fields[9:]
            if len(samples) >= 2:
                out.write("\t".join(fields[:11] + ["GENO3"]) + "\n")
            elif len(samples) == 1:
                out.write("\t".join(fields[:10] + ["GENO2", "GENO3"]) + "\n")
            else:
                out.write("\t".join(fields[:9] + ["GENO", "GENO2", "GENO3"]) + "\n")
            continue
        chrom = line.split("\t", 1)[0]
        key = chrom
        if key in targets and counts[key] < 20:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 11:
                out_fields = fields[:11] + [synthetic_sample(fields[10], fields[8], line_index, 2)]
            elif len(fields) == 10:
                out_fields = fields[:10] + [
                    synthetic_sample(fields[9], fields[8], line_index, 1),
                    synthetic_sample(fields[9], fields[8], line_index, 2),
                ]
            else:
                continue
            out_fields[7] = strip_conflicting_info(out_fields[7])
            out.write("\t".join(out_fields) + "\n")
            counts[key] += 1
            kept += 1
            line_index += 1
        if kept >= 60:
            break
PY

bgzip -c "${T}/genotype.vcf" > "${T}/genotype.vcf.gz"
tabix -f -p vcf "${T}/genotype.vcf.gz"
printf '%s\n' "${T}/genotype.vcf.gz" > "${T}/genotype_files.txt"
bgzip -c "${T}/genotype.vcf" > "${T}/dbsnp.vcf.gz"
tabix -f -p vcf "${T}/dbsnp.vcf.gz"
bcftools query -f'%CHROM\t%POS\t%ID\t%REF\t%ALT\n' "${T}/genotype.vcf.gz" | \
awk 'BEGIN { OFS = "\t" } {
    end_pos = $2 + ((length($4) > length($5)) ? length($4) : length($5)) - 1
    print $1, $2, end_pos, $3
}' | bgzip -c > "${T}/dbsnp.variants.gz"
tabix -f -s 1 -b 2 -e 3 "${T}/dbsnp.variants.gz"

python - <<'PY' "${T}/genotype.vcf" "${T}/genotype_reference.fa"
import sys
from collections import defaultdict

vcf_path, fasta_path = sys.argv[1], sys.argv[2]
max_len = defaultdict(int)
variants = defaultdict(list)

with open(vcf_path) as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        chrom, pos, _id, ref, _alt, *_rest = line.rstrip("\n").split("\t")
        pos = int(pos)
        end = pos + len(ref) - 1
        if end > max_len[chrom]:
            max_len[chrom] = end
        variants[chrom].append((pos, ref))

with open(fasta_path, "w") as out:
    for chrom in sorted(max_len):
        seq = ["N"] * max_len[chrom]
        for pos, ref in variants[chrom]:
            start = pos - 1
            seq[start:start + len(ref)] = list(ref)
        out.write(f">{chrom}\n")
        seq_str = "".join(seq)
        for idx in range(0, len(seq_str), 60):
            out.write(seq_str[idx:idx + 60] + "\n")
PY
samtools faidx "${T}/genotype_reference.fa"

link_into_test "${MWE_DIR}/AC.tmm_cpm_voom.expression.bed.gz" "${T}/AC.tmm_cpm_voom.expression.bed.gz"
link_into_test "${MWE_DIR}/AC.tmm_cpm_voom.expression.bed.gz.tbi" "${T}/AC.tmm_cpm_voom.expression.bed.gz.tbi"

python - <<'PY' "${T}/genotype.vcf" "${T}/sample_to_genotype_lookup.tsv" "${T}/sample_participant_lookup.tsv"
import sys

vcf_path, genotype_out_path, participant_out_path = sys.argv[1], sys.argv[2], sys.argv[3]
samples = []
with open(vcf_path) as fh:
    for line in fh:
        if line.startswith("#CHROM"):
            fields = line.rstrip("\n").split("\t")
            samples = fields[9:12]
            break

rows = list(zip(["sample1", "sample2", "sample3"], samples[:3]))

with open(genotype_out_path, "w") as out:
    out.write("sample_id\tgenotype_id\n")
    for sample_id, genotype_id in rows:
        out.write(f"{sample_id}\t{genotype_id}\n")

with open(participant_out_path, "w") as out:
    out.write("sample_id\tparticipant_id\tgenotype_id\n")
    for sample_id, genotype_id in rows:
        out.write(f"{sample_id}\t{genotype_id}\t{genotype_id}\n")
PY

python - <<'PY' "${T}/sample_participant_lookup.tsv" "${T}/modular_sos_hg_covariates.cov.gz"
import csv
import gzip
import sys

lookup_path, cov_path = sys.argv[1:]
with open(lookup_path) as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))
sample_ids = [row["genotype_id"] for row in rows]

with gzip.open(cov_path, "wt") as out:
    out.write("#id\t" + "\t".join(sample_ids) + "\n")
    out.write("cov_const\t" + "\t".join("0" for _ in sample_ids) + "\n")
PY

cat > "${T}/modular_sos_hidden_cov.tsv" <<'EOF'
#id	sample1	sample2
cov1	0	1
cov2	1	0
cov3	0.5	1.5
EOF
bgzip -f "${T}/modular_sos_hidden_cov.tsv"
mv -f "${T}/modular_sos_hidden_cov.tsv.gz" "${T}/modular_sos_hidden_cov.gz"

cat > "${T}/modular_sos_hidden_pheno.bed" <<'EOF'
#chr	start	end	gene_id	sample1	sample2
chr1	10000	10001	gene_chr1_1	1.0	2.0
chr2	20000	20001	gene_chr2_1	0.5	1.5
chr22	22000	22001	gene_chr22_1	1.2	0.8
EOF
bgzip -f "${T}/modular_sos_hidden_pheno.bed"
tabix -f -p bed "${T}/modular_sos_hidden_pheno.bed.gz"

POS_FILE="${T}/modular_sos_hg_synthetic.positions.tsv"
awk '
($1 == 1 || $1 == "chr1") {
    print $1 "\t" $4
    ++c
    if (c == 3) exit
}
' "${T}/AC.unrelated.plink_qc.prune.bim" > "${POS_FILE}"
if [[ ! -s "${POS_FILE}" ]]; then
    awk 'NR <= 3 { print $1 "\t" $4 }' "${T}/AC.unrelated.plink_qc.prune.bim" > "${POS_FILE}"
fi

awk '
NR == FNR {
    ids[++n] = $2
    next
}
header_printed == 0 {
    printf "#chr\tstart\tend\tgene_id"
    for (i = 1; i <= n; ++i) printf "\t" ids[i]
    printf "\n"
    header_printed = 1
}
{
    chrom = $1
    sub(/^chr/, "", chrom)
    pos = $2 + 0
    printf "chr%s\t%d\t%d\tgene_chr%s_%d", chrom, pos, pos + 1, chrom, FNR
    for (i = 1; i <= n; ++i) printf "\t%.3f", FNR + ((i - 1) % 11) / 10
    printf "\n"
}
' "${T}/AC.unrelated.plink_qc.prune.fam" "${POS_FILE}" > "${T}/modular_sos_hg_synthetic.expression.bed"
bgzip -f "${T}/modular_sos_hg_synthetic.expression.bed"
tabix -f -p bed "${T}/modular_sos_hg_synthetic.expression.bed.gz"
