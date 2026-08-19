#!/usr/bin/env Rscript
# Unit check for canonicalize_mirror_event_ids (pure function).
# Sources the worker (CLI is guarded) and exercises the mirror-pair rule on a
# tiny in-memory fixture. Prints "UNIT_OK" on success; stop()s on any mismatch.
args <- commandArgs(trailingOnly = TRUE)
worker <- args[1]
source(worker)

# Fixture: one mirror pair (INS A>AT + its swap AT>A) at 100; one non-mirror
# indel (C>CA, no swap) at 200; one SNP (G>T) at 300.
chrom <- c("22","22","22","22")
pos   <- c(100L,100L,200L,300L)
id    <- c("chr22:100:A:AT","chr22:100:AT:A","chr22:200:C:CA","chr22:300:G:T")
ref   <- c("A","AT","C","G")
alt   <- c("AT","A","CA","T")

em <- canonicalize_mirror_event_ids(chrom, pos, id, ref, alt)

# Only the mirror pair (2 rows) should survive.
stopifnot(nrow(em) == 2)
stopifnot(all(em$ID %in% c("chr22:100:A:AT","chr22:100:AT:A")))
# INS keeps pos, DEL is pos+1; single-base event "T".
ins <- em[em$event_type == "INS", ]
del <- em[em$event_type == "DEL", ]
stopifnot(nrow(ins) == 1, nrow(del) == 1)
stopifnot(ins$event_id == "chr22:100:INS:T")
stopifnot(del$event_id == "chr22:101:DEL:T")
# Non-mirror indel and SNP must be absent.
stopifnot(!any(grepl("200", em$event_id)))
stopifnot(!any(em$ID == "chr22:300:G:T"))

# Empty input -> 0-row result (no error).
em0 <- canonicalize_mirror_event_ids(character(0), integer(0),
                                     character(0), character(0), character(0))
stopifnot(nrow(em0) == 0)

cat("UNIT_OK\n")
