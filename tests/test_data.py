from pathlib import Path

import pandas as pd

from cla.data.assemble import (
    MATURE_MOTIF,
    AlleleResolver,
    build_chainspecs,
    load_specs,
    load_vdjdb,
    save_specs,
)
from cla.layout.spec import Role

LEADER = "MAVMAPRTLLLLLSGALALTQTWA"
MATURE_A2 = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKA"
MATURE_A3 = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKA"

# Entries are ordered as IMGT orders them: lowest-numbered allele of a group first,
# with a null variant (L suffix) ahead of the next real allele.
FASTA = f""">HLA:HLA00005 A*02:01:01:01 365 bp
{LEADER}{MATURE_A2}
>HLA:HLA00006 A*02:01:01:02L 365 bp
{LEADER}GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAXX
>HLA:HLA00007 A*03:01:01:01 365 bp
{LEADER}{MATURE_A3}
"""

VDJDB_COLS = ["cdr3.alpha", "cdr3.beta", "species", "mhc.a", "mhc.class", "antigen.epitope"]
VDJDB_ROWS = [
    # kept
    ["CAVRDNYGQNFVF", "CASSIVGGNEQFF", "HomoSapiens", "HLA-A*02:01", "MHCI", "GILGFVFTL", "3"],
    # kept by the loader; the 2-field name resolves through the group default
    ["CAVRDNYGQNFVF", "CASSIVGGNEQFF", "HomoSapiens", "HLA-A2", "MHCI", "GILGFVFTL", "2"],
    # loader filters: unpaired, mouse, class II, no epitope
    ["", "CASSIVGGNEQFF", "HomoSapiens", "HLA-A*02:01", "MHCI", "GILGFVFTL", "1"],
    ["CAVRDNYGQNFVF", "CASSIVGGNEQFF", "MusMusculus", "H-2Kb", "MHCI", "SIINFEKL", "1"],
    ["CAVRDNYGQNFVF", "CASSIVGGNEQFF", "HomoSapiens", "HLA-DRA*01", "MHCII", "GILGFVFTL", "1"],
    ["CAVRDNYGQNFVF", "CASSIVGGNEQFF", "HomoSapiens", "HLA-A*02:01", "MHCI", "", "1"],
    # builder drops: bad TCR residue, allele absent from the FASTA
    ["CAV*DNYGQNFVF", "CASSIVGGNEQFF", "HomoSapiens", "HLA-A*02:01", "MHCI", "GILGFVFTL", "1"],
    ["CAVRDNYGQNFVF", "CASSIVGGNEQFF", "HomoSapiens", "HLA-B*99:99", "MHCI", "GILGFVFTL", "1"],
]


def _write_vdjdb(tmp_path: Path) -> Path:
    path = tmp_path / "vdjdb_full.txt"
    cols = [*VDJDB_COLS, "vdjdb.score"]
    pd.DataFrame(VDJDB_ROWS, columns=cols).to_csv(path, sep="\t", index=False)
    return path


def _resolver(tmp_path: Path) -> AlleleResolver:
    path = tmp_path / "hla_prot.fasta"
    path.write_text(FASTA)
    return AlleleResolver.from_fasta(path)


def test_allele_variants_resolve_to_one_trimmed_entry(tmp_path):
    r = _resolver(tmp_path)
    variants = ["HLA-A*02:01", "HLA-A*02:01:01:01", "A*02:01", "HLA-A*02", "HLA-A2"]
    resolved = {r.resolve(v) for v in variants}
    assert resolved == {MATURE_A2}
    assert MATURE_MOTIF.match(MATURE_A2)
    # The null variant listed second must not shadow the real allele.
    assert r.fallback_trims == []

    assert r.resolve("HLA-B*99:99") is None
    assert r.unresolved["HLA-B*99:99"] == 1


def test_assembly_order_and_drop_counts(tmp_path):
    df = load_vdjdb(_write_vdjdb(tmp_path))
    assert [n for _, n in df.attrs["filters"]] == [8, 7, 6, 5, 4, 4]

    specs, dropped = build_chainspecs(df, _resolver(tmp_path))
    assert dict(dropped) == {"invalid_tcra": 1, "unresolved_allele": 1}
    assert len(specs) == 2 and df["vdjdb.score"].tolist()[:2] == [3, 2]
    for s in specs:
        assert [c.role for c in s.chains] == [
            Role.TCRA,
            Role.TCRB,
            Role.PEPTIDE,
            Role.MHC_HEAVY,
            Role.B2M,
        ]
        assert s.label == 1 and s.mhc_allele == "HLA-A*02:01"

    path = tmp_path / "specs.jsonl"
    save_specs(specs, path)
    assert load_specs(path) == specs
