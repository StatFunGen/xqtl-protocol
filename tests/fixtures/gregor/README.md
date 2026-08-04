# GREGOR downsized EUR reference fixture

A ~200 KB (8 KB gzipped) slice of GREGOR's ~20 GB EUR LD reference, carved so the
enrichment notebook (`code/SoS/enrichment/gregor.ipynb`) can be smoke-tested in CI.
GREGOR runs against it to completion in ~4 s and emits a valid `StatisticSummaryFile`.

## Contents
- `EUR/` — the reference GREGOR reads (`REF_DIR=tests/fixtures/gregor`, `POPULATION=EUR`):
  - `CHR22.db` — SQLite `CUBE` table, only the 30 positions of the 4 test index SNPs'
    cubes (index SNPs + their matched control candidates).
  - `CHR1.db`..`CHR21.db` — empty but valid `CUBE` dbs (the Makefile requires all 22 to
    exist; find.neighbors.LDbuddy.pl `exit(1)`s on a missing one).
  - `chr22.maf.dist.ldnum.id.txt` — the 4 index SNPs' cube-membership rows; `chr1..21`
    are header-only (required as Makefile prerequisites).
  - `cube.id.txt` — pruned from 21 MB to the 11 maf/dist/ldnum rows the index SNPs hit.
  - `reference.LD.buddy.threshold.txt` — copied verbatim.
  - `snp.db` is intentionally ABSENT: index SNPs are `chr:pos` (hg19), so the 7.3 GB
    rsID database is never opened.
- `index.snps.txt`, `test_peaks.bed` — committed test inputs (bed overlaps 2 of the 4
  index SNPs; must be run with `BEDFILE_IS_SORTED = true`).

## Provenance
Full EUR reference: https://csg.sph.umich.edu/GREGOR/index.php/site/download
(`GREGOR.EUR.ref.r2.greater.than.0.7.tar.gz.part.0{0,1}`, r2>=0.7, 1 Mb window, hg19).

## How to regenerate `EUR/`

Dev-only — the full reference is not available in CI. Download and reassemble the two
EUR parts (`cat …part.00 …part.01 | tar xzf -` → an `EUR/` dir), then run the steps
below against it. `sqlite3` must be available on your `PATH`.

The slice stays self-consistent because:
- Index SNPs are `chr:pos`, so `annotate.index.snp.pl` never opens `snp.db` (rsID-only path).
- GREGOR draws MAF/distance/LD-matched controls from each index SNP's OWN cube (a row of
  `chr22.maf.dist.ldnum.id.txt`); we keep only those cubes, and put exactly their member
  SNPs into `CHR22.db` (find.neighbors.LDbuddy.pl queries it `WHERE POS=?` per control).
- `cube.id.txt` is an EXACT-match value→id lookup: keep only the rows the index SNPs hit.
  `maf` is 1:1 with its id (select by id); `dist`/`ldnum` ids are binned, so select those
  by value — and `CHR22.db`'s `DIST`/`LDNUM` columns hold exactly the integer values
  annotate uses (no float-string matching).
- The Makefile lists chr1..22 unconditionally and find.neighbors.LDbuddy.pl `exit(1)`s on
  a missing `CHR<i>.db`, so chr1..21 get empty (valid) dbs and header-only cube files.

Steps (`SRC` = full `EUR/` dir, `OUT` = `tests/fixtures/gregor/EUR`):

```bash
SRC=~/Downloads/gregor_ref/EUR
OUT=tests/fixtures/gregor/EUR
SQLITE="sqlite3"
SCHEMA='CREATE TABLE CUBE (POS INT PRIMARY KEY NOT NULL,MAFID INT NOT NULL,DISTID INT NOT NULL,LDNUMID INT NOT NULL,MAF FLOAT NOT NULL,DIST INT NOT NULL,LDNUM INT NOT NULL,LDS TEXT NOT NULL);'
INDEX_POS="19449483 41441787 39507863 29485708"   # 4 chr22 index SNPs (hg19),
                                                  # each first member of a >=4-member cube
rm -rf "$OUT"; mkdir -p "$OUT"

# 1. Each index SNP's cube row (members are tab/pipe-delimited, allow either on both sides).
for p in $INDEX_POS; do
    grep -m1 -P "[\t|]22:${p}([\t|]|\$)" "$SRC/chr22.maf.dist.ldnum.id.txt"
done | sort -u > /tmp/cubes.tsv

# 2. Positions CHR22.db must hold = every member SNP of those cubes.
cut -f4 /tmp/cubes.tsv | tr '|' '\n' | sed 's/^22://' | sort -un > /tmp/pos.txt

# 3. CHR22.db: schema + just those rows, copied from the full db, then vacuumed.
POSLIST=$(paste -sd, /tmp/pos.txt)
$SQLITE "$OUT/CHR22.db" "$SCHEMA"
$SQLITE "$OUT/CHR22.db" "ATTACH '$SRC/CHR22.db' AS s; INSERT INTO CUBE SELECT * FROM s.CUBE WHERE POS IN ($POSLIST); VACUUM;"

# 4. chr22 cube-membership file: header + only our cubes. chr1..21: header only.
HDR=$(head -1 "$SRC/chr22.maf.dist.ldnum.id.txt")
{ printf '%s\n' "$HDR"; cat /tmp/cubes.tsv; } > "$OUT/chr22.maf.dist.ldnum.id.txt"
for i in $(seq 1 21); do printf '%s\n' "$HDR" > "$OUT/chr$i.maf.dist.ldnum.id.txt"; done

# 5. Empty (valid) 512-byte-page CUBE dbs for chr1..21.
for i in $(seq 1 21); do $SQLITE "$OUT/CHR$i.db" "PRAGMA page_size=512; $SCHEMA VACUUM;"; done

# 6. cube.id.txt: 3 header lines + maf (by id) and dist/ldnum (by value) rows the SNPs hit.
IDX=$(echo $INDEX_POS | tr ' ' ',')
MAFIDS=$(cut -f1 /tmp/cubes.tsv | paste -sd, -)
DISTVALS=$($SQLITE "$OUT/CHR22.db" "SELECT DIST FROM CUBE WHERE POS IN ($IDX);" | paste -sd, -)
LDNVALS=$($SQLITE "$OUT/CHR22.db" "SELECT LDNUM FROM CUBE WHERE POS IN ($IDX);" | paste -sd, -)
awk -F'\t' -v m="$MAFIDS" -v d="$DISTVALS" -v l="$LDNVALS" '
  BEGIN { split(m,ma,","); for(i in ma) MAF[ma[i]]=1;
          split(d,da,","); for(i in da) DIST[da[i]]=1;
          split(l,la,","); for(i in la) LDN[la[i]]=1 }
  FNR<=3 { print; next }
  $1=="maf"   && ($4 in MAF)  { print; next }
  $1=="dist"  && ($2 in DIST) { print; next }
  $1=="ldnum" && ($2 in LDN)  { print }
' "$SRC/cube.id.txt" > "$OUT/cube.id.txt"

# 7. LD-buddy r2 threshold marker (tiny; copied verbatim).
cp "$SRC/reference.LD.buddy.threshold.txt" "$OUT/reference.LD.buddy.threshold.txt"
```
