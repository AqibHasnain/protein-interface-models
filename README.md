# Chain-Layout Nuisance Sensitivity in Concatenation-Based Protein Interface Models

Sequence models of multi-chain complexes (TCR-pMHC, antibody-antigen, peptide-MHC)
are built by concatenating the component chains into one string and running a
protein language model over it. The relative-position geometry the model sees
therefore depends on how long each chain happens to be: a CDR3-beta three
residues longer shifts every downstream peptide-to-MHC offset by three, with no
change to the underlying biology. Structure models fixed this years ago by
inserting a large residue-index offset at chain breaks. Sequence-level interface
models mostly did not.

Phase 1 measures whether that omission leaks. Phase 2 implements the fix and
measures what it buys. Both ran, both on this laptop, for $0.00.

## What was found

ESM-2 at 150M and 650M is genuinely sensitive to content-free position-index
changes at chain breaks, and real receptors do not produce changes large enough
to trigger it.

Adding a constant gap to the position indices at chain breaks, with every token
bitwise identical, moves the MHC score 4.4 to 11.7 times as far as twenty real
amino-acid substitutions, on an allele-balanced sample of 51 HLA alleles. That
holds at both model sizes whose estimator passes a pre-set sanity check. Peptide
and CDR3-beta fail that check at every size and support no claim.

The gap has to reach 25 index steps for the effect to appear. Natural CDR3-beta
length varies by 3 to 4 residues, and matched complexes differing only in that
length show no length-attributable effect on the score. Trained directly on
binding, the chain-aware module does not beat a parameter-matched baseline on
either held-out shard.

The recommendation is against adopting a chain-aware positional module for this
setting. `docs/REPORT.md` carries the full results, the four claims that were
withdrawn along the way, and fourteen limitations.

## Reproducing

```bash
make setup     # uv venv + pinned deps
make data      # download VDJdb and IPD-IMGT/HLA, assemble complexes (idempotent)
make audit     # score, perturb, rescore, report GSI with cluster-bootstrap CIs
make report    # regenerate every figure and table from committed results
```

`make test` runs the full suite, including the leakage test, on CPU.

## The measurement

A **pure-geometry operator** holds `input_ids` bitwise identical and changes only
`position_ids`. ESM-2 uses rotary embeddings and no absolute position embeddings,
so this is exactly the nuisance axis of interest with no token-level confound.
Nullness is `torch.equal` on the input ids rather than an argument in prose, and
every operator has a test asserting it.

The headline number is the **Geometry Sensitivity Index**:

```
GSI = median |score shift from the relayout| / median |score shift from 20 real substitutions|
```

GSI = 1.0 means a content-free change in position indices moves the score as far
as twenty real amino-acid substitutions. Intervals are percentile bootstraps
resampling MHC alleles, not examples: the raw sample is dominated by one allele
and example-level intervals are far too narrow.

An earlier design used token-inserting operators and normalized by the
binder/non-binder gap. Both were withdrawn. `docs/DECISIONS.md` records why.

## Compute

Everything runs locally on Apple Silicon through PyTorch's MPS backend and CPU.
No cloud account, no credentials, no spend. See `docs/COMPUTE_LOG.md`.

## Layout

```
src/cla/
  data/      sources, assembly from allele name to chain sequence, splits
  layout/    ChainSpec, naive concatenation, the geometry and perturbation operators
  model/     ESM-2 loading, span pseudo-log-likelihood, the chain-aware positional module
  train/     LoRA arms, the binding classifier, gates
  eval/      metrics, the audit driver, cluster bootstrap, figures
  utils/     seeding, run manifest, structured logging
configs/     splits.yaml is the single source of truth for held-out data
docs/        DECISIONS.md, COMPUTE_LOG.md, AUDIT.md, REPORT.md
results/     committed metrics JSON and regenerable figures
```

Notebooks contain no logic. Everything testable lives in `src/`.

## What this does not claim

No improved TCR design, no state-of-the-art binding prediction, no wet-lab
validation. Class I only, one backbone family, CDR3 loops rather than
full-length receptor chains, mismatched decoys rather than validated
non-binders, and a backbone that is untrained for binding.
