# Chain-layout sensitivity in concatenation-based protein interface models: Phase 1 report

## Summary

For ESM-2 150M and 650M scoring the MHC heavy chain inside a TCR-pMHC complex, adding a constant gap
to the position indices at chain breaks, with every token bitwise identical, moves the score 4.8 to
10.5 times as far as twenty real amino-acid substitutions (GSI; cluster-bootstrap lower bounds 1.96
to 9.78). These are the only two of nine model-target conditions whose estimator passes its pre-set
sanity check. Peptide and CDR3-beta fail it at every size and support no claim. The backbone is
untrained for binding. Gate G1 reads PROCEED on the admissible MHC conditions alone. Four earlier
claims were withdrawn (section 6).

Phase 2 then found that the artifact does not occur at the scale real receptors produce. Matched
complexes differing only in CDR3-beta length show no length-attributable effect on the MHC score
(excess over a shuffled-length null 0.0003 nats per residue, interval [-0.0013, 0.0018]). Natural
CDR3-beta length varies by 3 to 4 residues, which falls in the small-gap regime where Phase 1 already
measured GSI at or below 1.8. The sensitivity is real and large at offsets of 25 or more, and
receptors do not generate offsets that large. On this evidence a reader deciding whether to adopt a
chain-aware positional module should decline. Section 8 gives the Phase 2 results in full.

## 1. Motivation and framing

Structure predictors insert a large residue-index offset at chain breaks. Sequence-level interface
models mostly concatenate chains and run a protein language model over the result, so the relative
offset between peptide and MHC depends on the length of whatever chain sits between them. A CDR3-beta
three residues longer shifts every downstream offset by three with no change in biology.

The project follows Spatial Transformer Networks (Jaderberg et al., 2015): name a nuisance
transformation group that an architecture handles by accident, measure the damage, and fix it with a
drop-in module that leaves the loss, optimizer and parameter count unchanged. The nuisance group is
chain relayout. The Phase 2 module is per-chain position reset, a chain-identity embedding and a large
relative-position offset across breaks. Phase 1 decides whether it is worth building.

## 2. Pre-registered claims

- H1 (audit): a biologically null relayout changes the binding score by a non-trivial amount relative
  to the binder/non-binder separation.
- H2 (fix): restoring chain-relative position structure reduces that sensitivity without hurting
  held-out generalization on strict allele and epitope splits.
- H3 (mechanism): the H2 gain concentrates where chain lengths deviate from the training mode.

Phase 1 addresses H1 only. The pre-registered operators were P1 (permute chains), P2 (separator
count), P3 (native flank padding), P4 (neutral spacer) and P5 (length-changing positive control),
with NSI = median |score change| / |median binder - median non-binder|. PROCEED required NSI >= 0.25
and Spearman rho <= 0.90 for two operators on two models; STOP was NSI < 0.15 everywhere.

Three things changed, each logged in `docs/DECISIONS.md` when it happened: the NSI denominator proved
unresolvable (4.1); P2 and P4 proved to measure token insertion and were replaced by position-only
operators (section 6); and the targets and admissibility check were added after pilot data had been
seen. The replacement metrics were not pre-registered, and this is not a blind pre-registration. The
0.25 admissibility threshold was fixed in code before the three-model results were read and has not
moved. The GSI gate rule (GSI >= 1.0 on at least one target with cluster-bootstrap lower bound > 0.5,
for two in-distribution gaps on two models) was written after the 35M pilot.

## 3. Method

**Data.** VDJdb release 2026-06-03, human, class I, paired CDR3 alpha and beta (83,886 of 192,753
rows); IPD-IMGT/HLA 3.65.0; beta-2 microglobulin UniProt P61769 residues 21-119. After dropping 125
unresolved-allele rows and deduplicating on chain identity, 76,524 complexes remain, each with
peptide, MHC heavy chain trimmed to residues 1-180, beta-2 microglobulin, CDR3-alpha and CDR3-beta.
Total length is 305 to 332 (median 317). Alleles and epitopes are held out by membership test; a
held-out epitope removes every TCR seen against it (`data/SOURCES.md`). Decoys are mismatched pairs,
not validated non-binders.

**Models and runs.** ESM-2 35M, 150M, 650M, float32, one Apple M4 Max (MPS), $0.00
(`docs/COMPUTE_LOG.md`). 35M and 150M: n = 1000, gaps 0, 1, 2, 5, 10, 25, 50, 100, 150. 650M: n = 250,
gaps 0, 1, 5, 25, 100. The score of a target span is its pseudo-log-likelihood given the rest of the
complex. PEPTIDE and TCRB are masked whole; MHC_HEAVY uses windowed masking (windows of 15, stride 15,
up to 4), because masking 180 residues at once leaves the model worse than chance.

**Pure-geometry operator.** ESM-2 positions enter only through rotary embeddings, and `EsmModel`
accepts `position_ids`. Rotary attention depends on index differences, so only cross-chain offsets
matter. G1 adds a constant gap at every chain break; G2 adds it before the peptide only. `input_ids`,
`attention_mask` and length are unchanged, and a test asserts `torch.equal` on the token tensors. Gap 0
gives median shift exactly 0.0 in all nine model-target pairs for both operators
(`fig5_inertness.png`). Length is unchanged, so token dropout's mask-fraction rescaling is identical
before and after; every result is computed with token dropout on and off. Gaps are capped at 150
because ESM-2's trained position range is 1026.

**GSI.** median |score(gap g) - score(gap 0)| divided by median |score(x) - score(x with 20 real
substitutions in a context chain)|. GSI 1.0 means the content-free change moves the score as far as
twenty mutations. Raw nats accompany every ratio because the denominator is small.

**Admissibility.** The median baseline span score must exceed the uniform-prior score (-ln 20 =
-2.996) by 0.25 nats. A score at the prior carries almost no information, so its movement cannot
support a claim.

**Intervals.** 95% bootstrap, 10,000 resamples. MHC_HEAVY resamples alleles: the audited sample has 14
distinct alleles at 35M and 150M and 9 at 650M, HLA-A*02:01 is 69% of examples, and an example-level
bootstrap gives intervals 1.6 to 3.8 times too narrow. PEPTIDE and TCRB intervals are example-level
and carry no claim.

## 4. Results

### 4.1 No binding discrimination

Median binder-minus-decoy separation for the peptide is 0.0017, 0.0075 and 0.0088 nats at 35M, 150M
and 650M, so the pre-registered NSI denominator is unresolved and NSI is reported as not computable.

| Model | PEPTIDE AUROC | TCRB AUROC | MHC_HEAVY AUROC |
|---|---|---|---|
| 35M | 0.489 [0.464, 0.515] | 0.533 [0.507, 0.557] | 0.511 [0.486, 0.537] |
| 150M | 0.500 [0.475, 0.526] | 0.519 [0.494, 0.545] | 0.569 [0.544, 0.594] |
| 650M | 0.490 [0.439, 0.540] | 0.521 [0.470, 0.571] | 0.477 [0.426, 0.527] |

Token dropout on, example-level intervals (`fig4_discrimination.png`). With dropout off the highest
value is MHC_HEAVY at 150M, 0.616 [0.591, 0.640]; MHC intervals are too narrow and the span is scored
against mismatched decoys, so it is not read as binding signal. Zero-shot concatenated ESM-2 does not
discriminate TCR-pMHC binding at any size.

### 4.2 Admissibility

| Target | Estimator | 35M | 150M | 650M |
|---|---|---|---|---|
| PEPTIDE | whole span | 0.037 | 0.039 | 0.050 |
| TCRB | whole span | 0.096 | 0.104 | 0.089 |
| MHC_HEAVY | windowed | 0.068 | 0.627 | 1.016 |

Margin over the uniform prior, required 0.25; only MHC_HEAVY at 150M and 650M passes. A 200-spec
probe at 35M recorded in `docs/DECISIONS.md` found a dataset unigram model ahead of ESM-2 on CDR3-beta
and no masking estimator that lifts peptide or CDR3-beta to 0.25. Its outputs are not in `results/`
and its numbers are not reproduced here.

### 4.3 Headline

MHC_HEAVY, G1, token dropout on, cluster-bootstrap intervals (`mhc_cluster_ci.tsv`, `table1.md`,
`fig1_dose_response.png`, `fig3_absolute_scale.png`).

| Model | n | Gap | Shift (nats) | 20-substitution shift (nats) | GSI [95% CI] |
|---|---|---|---|---|---|
| 150M | 1000 | 1 | 0.032 | 0.0446 | 0.71 [0.68, 0.92] |
| 150M | 1000 | 5 | 0.101 | 0.0446 | 2.28 [1.81, 2.96] |
| 150M | 1000 | 25 | 0.383 | 0.0446 | 8.59 [6.77, 9.40] |
| 150M | 1000 | 100 | 0.469 | 0.0446 | 10.52 [9.78, 16.11] |
| 650M | 250 | 1 | 0.030 | 0.0356 | 0.84 [0.52, 2.29] |
| 650M | 250 | 5 | 0.066 | 0.0356 | 1.85 [1.15, 2.11] |
| 650M | 250 | 25 | 0.172 | 0.0356 | 4.83 [1.96, 5.66] |
| 650M | 250 | 100 | 0.181 | 0.0356 | 5.08 [2.39, 5.97] |

No claim is made at gaps 1 and 2. G2 stays below 1.0 for MHC_HEAVY at both sizes (0.50 at 150M and
0.33 at 650M, gap 25), so the effect needs the gap at every chain break.

### 4.4 Generalization across alleles

The main sample has Kish effective sample size 1.92, so the effect could have been a property of
HLA-A*02:01. An allele-balanced 150M resample (51 alleles, at most 27 each, n = 610, effective sample
size 26.86, baseline margin 0.687) agrees:

| Gap | Balanced GSI [95% CI] | Concentrated GSI [95% CI] |
|---|---|---|
| 1 | 0.50 [0.43, 0.59] | 0.71 [0.68, 0.92] |
| 5 | 1.79 [1.58, 2.04] | 2.28 [1.81, 2.96] |
| 25 | 6.96 [6.18, 7.85] | 8.59 [6.77, 9.40] |
| 100 | 11.71 [10.39, 13.19] | 10.52 [9.78, 16.11] |

The 650M balanced run had not completed at time of writing. The committed files hold no per-allele
GSI values.

### 4.5 Cross-chain content is used

Complex minus scrambled-context pseudo-log-likelihood (other chains' residues shuffled), MHC_HEAVY,
median nats, n = 300 (`context_gain.json`):

| Model | Dropout on | Dropout off |
|---|---|---|
| 35M | +0.028 | -0.093 |
| 150M | +0.599 | +0.612 |
| 650M | +0.190 | +0.216 |

At 150M and 650M the term is positive and agrees across arms. At 35M it changes sign, consistent
with an unusable estimator. The complex-minus-alone comparison is confounded by unequal mask
fractions at 650M (-0.03 on, +0.54 off) and is not used.

Cluster intervals over alleles are 3.5 to 13.7 times wider than the example-level ones and still
exclude zero in all four 150M and 650M cells. The tightest is 650M with dropout on: median 0.190,
interval [0.033, 0.217]. The artifact therefore acts on real cross-chain computation.

### 4.6 Scale

Across the two admissible sizes sensitivity falls: at gap 100 from 10.52 [9.78, 16.11] to 5.08
[2.39, 5.97] (non-overlapping), and from 0.469 to 0.181 nats. At gap 25 the intervals [6.77, 9.40]
and [1.96, 5.66] also do not overlap. About 5x remains at 650M (`fig2_scale.png`). Two points give a
direction and no scaling law.

### 4.7 Token dropout

| Model | Gap | GSI on | GSI off [95% CI] |
|---|---|---|---|
| 150M | 25 | 8.59 | 9.64 [8.95, 10.65] |
| 150M | 100 | 10.52 | 18.14 [16.84, 20.05] |
| 650M | 25 | 4.83 | 4.55 [3.91, 5.33] |
| 650M | 100 | 5.08 | 0.57 [0.48, 0.73] |

Off-arm intervals are example-level and too narrow for MHC_HEAVY. The 650M gap-100 value falls below
1.0 with dropout off, and that discrepancy was not investigated, so the robust 650M claim is the
gap-25 one, where the arms agree. The 150M results hold in both arms.

### 4.8 Peptide and CDR3-beta

GSI exceeds 1.0 in many conditions and reaches 3.3 to 3.7 at gap 150 (`table1.md`), with gap 0
exactly 0. These rows are drawn with hollow markers in `fig1_dose_response.png`. They are not
evidence: the underlying scores sit 0.04 to 0.10 nats from the uniform prior.

## 5. Gate G1

PROCEED. The driver (`aggregate.py`) returned PROCEED on PEPTIDE and TCRB for all three models. Those
targets fail admissibility at every size, so a gate resting on them would have failed its own
criteria. The verdict rests on MHC_HEAVY at 150M and 650M alone: GSI 4.83 to 10.52 at gaps 25 and
100, every cluster-bootstrap lower bound at or above 1.96 against a requirement of 0.5. The
pre-registered NSI gate had no computable verdict.

## 6. What was withdrawn

1. **Step versus graded response.** The 35M sweep showed the peptide G1 response flat from gap 1 and
   concluded the offset magnitude was not worth tuning. At 150M it rises to a peak near gap 25 and
   falls. The step was a 35M artifact. Phase 2 must sweep the offset, not fix it at 200.
2. **Sensitivity increases with scale.** MHC GSI at gap 150 going from 3.63 (35M) to 12.46 (150M) was
   recorded as the main finding. It compared an inadmissible estimator with an admissible one. With
   650M admitted the direction reverses (4.6).
3. **No cross-chain reasoning to corrupt.** True at 35M, where context gain was near zero. False at
   150M and 650M (4.5). The corrected claim is stronger.
4. **GSI claims at gaps 1 and 2.** They fell below 1.0 under the cluster bootstrap.

Also replaced: the pre-registered operators. On review, P2 and P4 moved the score mainly through
inserted `<pad>` separator tokens, whose embedding has an out-of-distribution norm that every layer
attends to. The P5 control moved scores less than the nulls it was to bound. P1 left chains adjacent
to special tokens. The position-only G operators replace them; the content-coupled P operators stay
in the result files under their own labels. The measurements behind these findings (35M, CPU) are in
the decision log and are not regenerated from `results/`.

## 7. Limitations

1. **Most of the audit is not evidence.** PEPTIDE and TCRB are inadmissible at every size. Two of
   nine conditions support a claim, both on the MHC heavy chain. The peptide and CDR3 spans, which
   decide binding, are unestablished.
2. **The admissibility rule may be wrong.** The uniform-prior reference and the 0.25 threshold are
   arguably miscalibrated for hypervariable spans, where a near-uniform masked-LM prior is expected.
   A dataset-unigram reference would be better. The rule was deliberately not changed after the GSI
   outcome was seen, because a criterion rewritten after learning what it excludes is a formality.
   The rule may therefore have disqualified two of three targets wrongly. Rerunning with a unigram
   reference, fixed before the rerun and evaluated blind, is the first item of future work.
3. **Untrained backbone.** Zero-shot ESM-2 does not discriminate binders. Nothing is shown about a
   fine-tuned interface model, which is the claim that would matter.
4. **650M is thin.** n = 250, 9 alleles, effective sample size 1.96, five gaps, and no completed
   balanced run. The scale direction rests on the smallest sample, and its gap-100 value depends on
   token dropout.
5. **Allele concentration.** Effective sample size 1.92 in the main runs. The balanced run covers 150M
   only and reports no per-allele values.
6. **The cluster intervals are themselves optimistic.** A percentile bootstrap over 9 to 14 clusters,
   one of which is 69% of the sample, has poor coverage. The data supports the statement that G1 at
   gaps 25 and 100 sits well above 1.0 at 150M and 650M with token dropout on. It does not support
   the specific bounds, and the example-level intervals it replaced were worse still. The
   allele-balanced run at 51 alleles is the stronger evidence and exists only at 150M.
7. **TCR chains are CDR3 loops only.** VDJdb has no full variable domains, so TCR length variation is
   far smaller than in a real receptor, and operator P3 is not applicable.
8. **Scope.** MHC trimmed to alpha-1/alpha-2, class I only, one backbone family. Class II,
   antibodies and other architectures are untested.
9. **Decoys are mismatched pairs.** They are not validated non-binders. The chance-level AUROC
   belongs to the model, since real negatives would not give an untrained ESM-2 a signal.
10. **Metrics defined after pilot data.** GSI, the admissibility check and NSI-C followed the
   60-example 35M pilot.
11. **Estimator coverage.** MHC_HEAVY windowed scoring covers at most 60 of 180 residues.
12. **No wet-lab validation,** and no claim about TCR design or binding accuracy.
13. **Unverifiable numbers.** The unigram comparison, the 35M defect measurements and the pilot
    tables exist only in the decision log rather than in `results/`.

## 8. Phase 2: the fix, and why it is not warranted on this evidence

Gate G1 passed, so Phase 2 ran. Compute was local and free, so it ran at the plan's own dev-loop
scale (section 7.3) rather than the budgeted 650M runs.

### 8.1 The module

`src/cla/model/positional.py`, 131 lines against the plan's 150-line ceiling. `ChainAwarePositions`
exposes per-chain index reset, a learned chain-identity embedding and a chain-break offset, each
independently switchable. ESM-2 takes `position_ids` and positions enter only through rotary
embeddings, so reset and offset are pure index remaps with no parameters, and only the
chain-identity embedding needs gradients. The embedding attaches by a forward hook on
`model.esm.embeddings` and initializes to zero, so it is an exact no-op before training. With every
flag off the module returns `position_ids=None` and the baseline path is untouched.

Asserted in `tests/test_positional.py`: with reset enabled, each chain's position indices are
identical under any chain order. Full embedding invariance under permutation does not hold, because
concatenation order still changes which chains a residue attends to, and the test says so rather
than claiming otherwise.

### 8.2 Two findings about the plan's own prescription

**The default offset of 200 does not fit.** With five chains, a 200-index break offset puts every
layout past ESM-2's trained range of 1026, leaving the arm with no in-distribution examples. Offset
100 already loses the gap-100 cell. On this backbone the usable range is roughly 50 or below.

**The module is not free at inference.** Applying it zero-shot degrades the score: reset alone moves
the median baseline by 0.66 nats, leaving a margin of 0.034 over the uniform prior against a 0.25
usability bar. Reset plus offset 25 falls below the uniform prior outright. ESM-2 has never seen
reset positions. Only A0 and offset-25 remain usable scorers.

### 8.3 The circular arm

Measuring per-chain reset against the Phase 1 gap operator returns exactly 0, because the operator
injects a position gap and reset discards incoming positions. That is an implementation check, not
evidence. It is reported as such and excluded from every claim.

### 8.4 Native chain-length variation: the test that matters

Complexes matched on allele and peptide, differing in CDR3-beta length; statistic is the median
within-set slope of MHC score against CDR3-beta length, against a null with lengths shuffled within
each set. 52 sets, 445 members, 28 alleles on the allele-balanced sample.

| Arm | Margin | Usable | Sensitivity (nats/residue) | Excess over null | Ratio vs A0 |
|---|---|---|---|---|---|
| A0 | 0.687 | yes | 0.0039 [0.0025, 0.0053] | 0.0003 [-0.0013, 0.0018] | 1 |
| A2 reset | 0.034 | no | 0.0026 [0.0017, 0.0035] | 0.0001 [-0.0006, 0.0008] | 0.68 [0.40, 1.17] |
| A3 offset 25 | 0.401 | yes | 0.0026 [0.0020, 0.0034] | 0.0003 [-0.0001, 0.0011] | 0.66 [0.42, 1.16] |
| A3 offset 50 | 0.222 | no | 0.0021 [0.0014, 0.0030] | -0.0003 [-0.0007, 0.0005] | 0.54 [0.34, 1.00] |
| A2+A3 offset 25 | -0.527 | no | 0.0025 [0.0017, 0.0038] | -0.0007 [-0.0017, 0.0009] | 0.65 [0.44, 1.11] |
| A2+A3 offset 50 | -0.114 | no | 0.0033 [0.0018, 0.0061] | -0.0004 [-0.0020, 0.0006] | 0.86 [0.43, 1.89] |

The baseline excess over null contains zero, so there is no length-attributable effect to remove.
Every ratio interval contains 1 except offset 50, which touches it. The measured slope is within-set
variation from TCR content, mostly CDR3-alpha. On an extended pool (720 sets) the arms do cut the raw
slope to 0.30 to 0.48 of A0, but the null falls by the same amount, so that is reduced total score
variation rather than a removed length effect.

### 8.5 Chain order permutation

Permutation reorders tokens, so it cannot be erased by construction in the way the injected gap was.
The baseline moves 0.561 nats [0.530, 0.590] under permutation and every arm reduces it: offset 25 to
0.49 of A0, offset 50 to 0.20. The reduction is real. Two qualifications: reset's ratio of 0.004 is
near-circular, since a reset layout presents the model with the same (token, position) multiset under
any order; and of the arms that reduce sensitivity most, only offset 25 is still a usable scorer.

### 8.6 Phase 2 verdict

The module does what it was designed to do on the permutation axis and costs score quality to do it.
On the axis that motivated the project, native chain-length variation, there is no baseline effect
for it to fix. The honest recommendation is against adopting it for this setting.

What would change that: full-length TCR variable domains instead of CDR3 loops, where length varies
by far more than 3 to 4 residues and could reach the offset regime where Phase 1 shows the model is
genuinely sensitive; and a backbone fine-tuned for binding, where the score carries signal worth
protecting.

## 9. Conclusion

Where the estimator passes its own check, ESM-2 treats a content-free change in position indices at
chain breaks as 4.8 to 10.5 times as large as twenty real substitutions. The effect holds across 51
alleles at 150M and acts on genuine cross-chain computation. Scale lowers it without removing it. H1
is supported for the one target where it can be tested.

H2 is not. The artifact needs an offset of 25 or more to appear, and the nuisance that motivated the
project, CDR3-beta length variation, moves offsets by 3 to 4. Phase 2 measured no length-attributable
effect in the baseline, so the module has nothing to fix on that axis, and the arms that most reduce
permutation sensitivity have degraded into unusable scorers. The recommendation is against adopting a
chain-aware positional module for this setting.

The two results together are the useful output: a real positional artifact, and the finding that
natural chain geometry does not reach it. Either one alone would have been misleading. Reporting only
the first would have motivated a fix nobody needs; reporting only the second would have missed a
genuine defect that full-length chains could expose.

Phase 2 ran at dev-loop scale and is reported in section 8. What a funded Phase 2 would still need
to establish:

1. That a model fine-tuned for binding shows the same sensitivity.
2. That the module lowers GSI and preserves or improves held-out AUROC on strict allele and epitope
   splits, with the offset swept and raw nats reported.
3. A binding-relevant target that passes admissibility, through a trained scorer or an
   admissibility reference defined before the run that suits hypervariable spans.
4. Validated non-binders in place of mismatched decoys, and allele-balanced evaluation at every size.
5. The H3 test of whether gains concentrate where chain lengths deviate from the training mode.

Artifacts: `results/phase1/` (run `nsi.json` and `summary.tsv`, `mhc_cluster_ci.tsv`,
`context_gain.json`, `gate_g1_gsi.json`, `figures/`), `results/phase2/` (`zeroshot_gsi.json`,
`noncircular_eval.json`, `training/`), `src/cla/model/positional.py`, `docs/DECISIONS.md`,
`docs/COMPUTE_LOG.md`, `data/SOURCES.md`.
