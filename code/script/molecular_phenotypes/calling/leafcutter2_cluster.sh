#!/usr/bin/env bash
# ============================================================
# leafcutter2_cluster.sh
# Cluster, classify and quantify junctions with LeafCutter2, for the leafcutter2_2
# and leafcutter2_cluster steps of splicing_calling_leafcutter2.ipynb.
#
# Three checks wrap the leafcutter2 call, because each failure they catch is
# otherwise silent or surfaces much later:
#   1. every junction file has the same column count. BED12 (regtools .junc) and
#      BED6 (star2junc) need different --offset values, so a mixed list cannot be right.
#   2. junction and annotation chromosome naming agree (chr22 against 22). If they
#      differ, nothing classifies and every junction simply comes back unannotated.
#   3. at least one cluster survived the read filters. If none did, clustering still
#      reports success and the failure appears later, in classification.
#
# --keepleafcutter1 is always passed so the LeafCutter1-format tables that
# leafcutter_ds.R reads are written alongside.
# ============================================================
# No pipefail: the checks read only the first lines (zcat | head), which ends zcat early.
set -eu

JUNCFILES=""
OUTPREFIX=""
RUNDIR=""
ANNOT=""
GENOME=""
MAXINTRONLEN=500000
MINCLUREADS=30
MINREADS=5
MINREADSTD=0.5
MINCLURATIO=0.001
OFFSET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --juncfiles)    JUNCFILES="$2"; shift 2 ;;
    --outprefix)    OUTPREFIX="$2"; shift 2 ;;
    --rundir)       RUNDIR="$2"; shift 2 ;;
    --annot)        ANNOT="$2"; shift 2 ;;
    --genome)       GENOME="$2"; shift 2 ;;
    --maxintronlen) MAXINTRONLEN="$2"; shift 2 ;;
    --minclureads)  MINCLUREADS="$2"; shift 2 ;;
    --minreads)     MINREADS="$2"; shift 2 ;;
    --minreadstd)   MINREADSTD="$2"; shift 2 ;;
    --mincluratio)  MINCLURATIO="$2"; shift 2 ;;
    --offset)       OFFSET="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

: "${JUNCFILES:?--juncfiles is required}"
: "${OUTPREFIX:?--outprefix is required}"
: "${RUNDIR:?--rundir is required}"
: "${ANNOT:?--annot is required}"
: "${GENOME:?--genome is required}"
: "${OFFSET:?--offset is required (0 for BED6, 1 for regtools .junc)}"

# 1. One offset applies to the whole run.
ncol=$(while read -r f; do [ -n "$f" ] && zcat -f "$f" | head -n1 | awk '{print NF}'; done < "$JUNCFILES" | sort -u | tr '\n' ' ')
if [ "$(echo $ncol | wc -w)" -gt 1 ]; then
  echo "ERROR: junction files mix formats (column counts: $ncol). A single --offset cannot suit both." >&2
  exit 1
fi

# 2. Chromosome naming.
jchr=$(zcat -f "$(head -n1 "$JUNCFILES")" | head -n1 | cut -f1)
achr=$(zcat -f "$ANNOT" | grep -v '^#' | head -n1 | cut -f1)
case "$jchr" in chr*) jp=prefixed;; *) jp=bare;; esac
case "$achr" in chr*) ap=prefixed;; *) ap=bare;; esac
if [ "$jp" != "$ap" ]; then
  echo "ERROR: chromosome naming differs. Junctions use '$jchr', annotation uses '$achr'." >&2
  exit 1
fi

leafcutter2 \
  --juncfiles "$JUNCFILES" \
  --outprefix "$OUTPREFIX" \
  --rundir "$RUNDIR" \
  --annot "$ANNOT" \
  --genome "$GENOME" \
  --maxintronlen "$MAXINTRONLEN" \
  --minclureads "$MINCLUREADS" \
  --minreads "$MINREADS" \
  --minreadstd "$MINREADSTD" \
  --mincluratio "$MINCLURATIO" \
  --offset "$OFFSET" \
  --keepleafcutter1

# 3. An empty result means the filters removed every cluster.
if [ "$(zcat "$RUNDIR/$OUTPREFIX.cluster_ratios.gz" | head -n2 | wc -l)" -lt 2 ]; then
  echo "ERROR: no clusters survived filtering. Lower --min-clu-reads (currently $MINCLUREADS)" >&2
  echo "  or --min-reads (currently $MINREADS), or check that the junction files are not empty." >&2
  exit 1
fi
