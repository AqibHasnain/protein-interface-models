# Data sources

Raw files live in `data/raw/` and are fetched by `cla.data.sources.ensure_all()`.
Assembled output is `data/processed/vdjdb_specs.jsonl`. All three sources were retrieved on 2026-10-10.

## VDJdb

- URL: https://github.com/antigenomics/vdjdb-db/releases/download/2026-06-03/vdjdb-2026-06-03.zip
- Version: release tag `2026-06-03` (file `vdjdb_full.txt`, extracted from the zip). Checksum pinned in `sources.py`.
- License: AGPL-3.0 (repository `LICENSE`).
- Filters (`assemble.load_vdjdb`), rows remaining after each:

| Filter | Rows |
|---|---|
| raw | 192753 |
| `species == HomoSapiens` | 176130 |
| `mhc.class == MHCI` | 170284 |
| `cdr3.alpha` and `cdr3.beta` both non-empty | 83886 |
| `antigen.epitope` non-empty | 83886 |
| `mhc.a` parses as an HLA allele | 83886 |

- Paired chains are required because the downstream complex has both TCRA and TCRB.
- Rows are not deduplicated. VDJdb has one row per clonotype per study and donor, so
  83761 specs contain 76525 distinct (role, sequence) identities. Deduplicate before
  making train/test splits.
- `vdjdb.score` (0-3) is kept in the loaded frame; 77133 of the 83886 rows have score 0.

## IPD-IMGT/HLA

- URL: https://raw.githubusercontent.com/ANHIG/IMGTHLA/37b324ba5d4fef45169c8b162c19bc851a1f8068/hla_prot.fasta
- Version: IPD-IMGT/HLA release 3.65.0, 2026-07-14, pinned to the release commit rather than the
  `Latest` branch (identical content at retrieval time). Checksum pinned in `sources.py`.
- License: see the IMGTHLA repository (https://github.com/ANHIG/IMGTHLA) and https://www.ebi.ac.uk/ipd/imgt/hla/about/licence/; not independently verified here.
- Processing (`AlleleResolver`):
  - Entries with expression suffixes (N, L, S, Q, C, A) are skipped, and allele names are reduced to two fields
    (the first listed entry per two-field name is used).
  - A one-field name such as `HLA-A*02` or `HLA-A2` resolves to the first two-field allele
    of that group in file order (`A*02:01`). This affects 14854 `HLA-A*02` rows plus other bare names.
  - Signal peptide is removed by cutting at the first `[GC]SHS[ML][RK]Y` match within the
    first 40 residues; no entry used by VDJdb needed the 24-residue fallback.
    The motif includes `K` at position 6 because C*01:02 and E alleles carry `MKY`/`LKY`.
  - The MHC chain is the full mature protein, including the transmembrane and cytoplasmic
    segments present in `hla_prot.fasta`; it is not trimmed to the ectodomain.
- Rows dropped for unresolved alleles: 125 (`HLA-A*08:01` x65, `HLA-A*24:01` x59, `HLA-B*12` x1),
  which have no matching entry in release 3.65.0.

## Beta-2 microglobulin

- URL: https://rest.uniprot.org/uniprotkb/P61769.fasta
- Version: UniProt P61769 as of 2026-10-10. The file is a moving target, so its checksum is
  recorded in `data/raw/OBSERVED_CHECKSUMS.json` on first download and verified afterward.
- License: CC BY 4.0.
- Residues 21-119 (mature chain, 99 aa) are hardcoded in `sources.py` and checked against the
  download. The same sequence is used for every spec.

## Assembled specs

- Result: 83761 ChainSpecs from 83886 filtered rows; 125 dropped as `unresolved_allele`,
  none dropped for invalid residues.
- TCRA and TCRB chains are CDR3 loops only; VDJdb has no full-length chains.
