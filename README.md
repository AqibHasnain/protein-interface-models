# Chain-Layout Nuisance Sensitivity in Concatenation-Based Protein Interface Models

Sequence models of multi-chain complexes (TCR-pMHC, antibody-antigen, peptide-MHC)
are built by concatenating the component chains into one string and running a
protein language model over it. The relative-position geometry the model sees
therefore depends on how long each chain happens to be: a CDR3-beta three
residues longer shifts every downstream peptide-to-MHC offset by three, with no
change to the underlying biology. Structure models fixed this years ago by
inserting a large residue-index offset at chain breaks. Sequence-level interface
models mostly did not.

**Phase 1 measures whether that omission leaks.** It is inference-only and costs
nothing. Phase 2 fixes it and measures the gain under matched conditions, and
only runs if Phase 1 returns a positive signal against a pre-registered
threshold.

## Reproducing Phase 1

```bash
make setup     # uv venv + pinned deps
make data      # download VDJdb and IPD-IMGT/HLA, assemble complexes (idempotent)
make audit     # score, perturb, rescore, report NSI with bootstrap CIs
```

`make test` runs the full suite, including the leakage test, on CPU.

## The measurement

A **null relayout** changes how a complex is written down without changing the
complex. The audit applies four of them (chain reordering, separator count,
native-flank restoration, neutral spacer insertion) plus a deliberately
non-null positive control, and asks how far the model's binding score moves.

The headline number is the **Nuisance Sensitivity Index**:

```
NSI = median |sPLL(x) - sPLL(perturb(x))|  /  |median sPLL(binders) - median sPLL(non-binders)|
```

NSI = 1.0 means a biologically meaningless relayout moves the score as much as
the entire binder/non-binder signal.

Every null operator has a unit test asserting that the multiset of
(role, sequence) pairs is unchanged. That assertion is what makes "this
perturbation is biologically null" a property of the code rather than a claim in
prose.

## Compute

Everything runs locally on Apple Silicon through PyTorch's MPS backend. No cloud
account, no credentials, no spend. See `docs/COMPUTE_LOG.md`.

## Layout

```
src/cla/
  data/      sources, assembly from allele name to chain sequence, splits
  layout/    ChainSpec, naive concatenation, the null relayout operators
  model/     ESM-2 loading, span pseudo-log-likelihood
  eval/      metrics, the audit driver, figures
  utils/     seeding, run manifest, structured logging
configs/     splits.yaml is the single source of truth for held-out data
docs/        DECISIONS.md, COMPUTE_LOG.md, AUDIT.md, REPORT.md
results/     committed metrics JSON and regenerable figures
```

Notebooks contain no logic. Everything testable lives in `src/`.

## What this does not claim

No improved TCR design, no state-of-the-art binding prediction, no wet-lab
validation. Class I only, one backbone family, CDR3 loops rather than
full-length receptor chains. `docs/REPORT.md` carries the full limitations
section.
