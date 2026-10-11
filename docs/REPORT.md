# Chain-layout sensitivity in concatenation-based protein interface models: Phase 1 report

## Summary

For ESM-2 150M and 650M scoring the MHC heavy chain inside a TCR-pMHC complex, adding a constant gap
to the position indices at chain breaks, with every token bitwise identical, moves the score 4.8 to
10.5 times as far as twenty real amino-acid substitutions (GSI; cluster-bootstrap lower bounds 1.96
to 9.78). These are the only two of nine model-target conditions whose estimator passes its pre-set
sanity check. Peptide and CDR3-beta fail it at every size and support no claim. The backbone is
untrained for binding. Gate G1 reads PROCEED on the admissible MHC conditions alone. Four earlier
claims were withdrawn (section 6).

Phase 2 then found two things. The chain-aware module does not produce an above-chance binding
predictor at either 35M or 650M, so H2 as written is not supported; paired by seed it does raise
held-out-epitope AUROC in all twelve arm-seed comparisons, from a baseline that sits below chance,
which is a direction worth one more experiment rather than a result. And the artifact does not occur
at the scale real receptors produce: matched complexes differing only in CDR3-beta length show no
length-attributable effect on the MHC score (excess over a shuffled-length null 0.0003 nats per
residue, interval [-0.0013, 0.0018]). Natural CDR3-beta length varies by 3 to 4 residues, which
falls in the small-gap regime where Phase 1 already measured GSI at or below 1.8. The sensitivity is
real and large at offsets of 25 or more, and receptors do not generate offsets that large. On this
evidence a reader deciding whether to adopt a chain-aware positional module should decline. Section
8 gives the Phase 2 results in full.

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
HLA-A*02:01. An allele-balanced resample (51 alleles, at most 27 each, n = 610, effective sample size
26.86) agrees at both sizes (`fig6_allele_balanced.png`). Intervals are cluster bootstraps over
alleles, token dropout on:

| Gap | 150M balanced | 650M balanced | 150M concentrated | 650M concentrated |
|---|---|---|---|---|
| 1 | 0.50 [0.40, 0.64] | 0.95 [0.69, 1.37] | 0.71 [0.68, 0.92] | 0.84 [0.52, 2.29] |
| 5 | 1.79 [1.49, 2.11] | 1.15 [0.87, 1.59] | 2.28 [1.81, 2.96] | 1.85 [1.15, 2.11] |
| 25 | 6.96 [6.20, 7.93] | 4.41 [3.70, 5.30] | 8.59 [6.77, 9.40] | 4.83 [1.96, 5.66] |
| 100 | 11.71 [10.45, 13.17] | 3.27 [2.65, 4.19] | 10.52 [9.78, 16.11] | 5.08 [2.39, 5.97] |

Dropping HLA-A*02:01 entirely moves no balanced point estimate by more than 4%, so the effect is not
carried by the dominant allele.

GSI is not monotone in gap at 650M: gap 100 scores below gap 25, where 150M keeps rising. A single
sample at two sizes does not separate a real non-monotonicity from noise, so the claim here is only
that both sizes exceed 1.0 by a wide margin at gap 25 and above.

Per-allele GSI (n = 27 each, `results/phase1/allele_balanced/per_allele_gsi.tsv`) has median 7.07 at
150M and 3.90 at 650M for gap 25, with ranges spanning 0.46 to 38.2 and 0.21 to 63.1. At that sample
size 22 of 51 alleles clear 1.0 on their own lower bound at 150M, 21 of 51 at 650M. The effect is
present across the allele panel and is not individually resolvable allele by allele.

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
and [1.96, 5.66] also do not overlap. About 5x remains at 650M (`fig2_scale.png`).

The allele-balanced samples give the same direction on 51 alleles rather than 9: gap 25 falls from
6.96 [6.20, 7.93] to 4.41 [3.70, 5.30], gap 100 from 11.71 [10.45, 13.17] to 3.27 [2.65, 4.19]. Both
pairs are non-overlapping. Two points give a direction and no scaling law, and the effect at 650M is
still several times the substitution scale.

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

5. **"The full module fails its mid-training gate."** True at 35M, where gate G2 killed A4 at step 30
   on a collapsed MHC margin. At 650M the same arm passes the gate and runs to completion. The gate
   fired correctly on the evidence it had; the claim about the module did not survive the larger
   backbone.
6. **"LoRA at this scale damages the scorer regardless of the module."** Drawn from every 35M arm
   including the baseline ending with an MHC margin at or below zero. The 650M baseline holds
   +0.751, so this was a property of a 35M backbone with no margin to spare.

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
4. **Phase 2 ran at dev-loop scale.** 300 steps per arm; the masked-LM arms are one seed each. H2
   is tested on the binding classifier at three seeds per arm (8.7), and the answer is negative, but
   with every AUROC interval covering 0.5 the test is underpowered as well as negative. A larger
   backbone, more steps or validated negatives could all change it.
5. **650M main sample is thin.** n = 250, 9 alleles, effective sample size 1.96, five gaps. The
   balanced run at 51 alleles supersedes it and reverses its gap-100 reading.
6. **Allele concentration.** Effective sample size 1.92 in the main runs. The balanced runs at 51
   alleles cover both sizes; no single allele is individually well resolved at n = 27.
7. **The cluster intervals are themselves optimistic.** A percentile bootstrap over 9 to 14 clusters,
   one of which is 69% of the sample, has poor coverage. The data supports the statement that G1 at
   gaps 25 and 100 sits well above 1.0 at 150M and 650M with token dropout on. It does not support
   the specific bounds, and the example-level intervals it replaced were worse still. The
   allele-balanced runs at 51 alleles are the stronger evidence and now exist at both sizes.
8. **TCR chains are CDR3 loops only.** VDJdb has no full variable domains, so TCR length variation is
   far smaller than in a real receptor, and operator P3 is not applicable.
9. **Scope.** MHC trimmed to alpha-1/alpha-2, class I only, one backbone family. Class II,
   antibodies and other architectures are untested.
10. **Decoys are mismatched pairs.** They are not validated non-binders. The chance-level AUROC
   belongs to the model, since real negatives would not give an untrained ESM-2 a signal.
11. **Metrics defined after pilot data.** GSI, the admissibility check and NSI-C followed the
   60-example 35M pilot.
12. **Estimator coverage.** MHC_HEAVY windowed scoring covers at most 60 of 180 residues.
13. **No wet-lab validation,** and no claim about TCR design or binding accuracy.
14. **Unverifiable numbers.** The unigram comparison, the 35M defect measurements and the pilot
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

### 8.6 Training the arms, and gate G2

300 steps at batch 16 over 4,800 examples, LoRA r=16 on attention and FFN, identical batches, masks,
optimizer and schedule across arms, trainable parameters matched by padding the arms that have no
chain embedding. One seed per arm. Run at 35M on CPU and at 650M on MPS, 29 minutes per arm.

The 650M runs are the informative ones and the 35M runs are reported because they came first and
because one of their conclusions turned out to be an artifact of the size.

| Arm | Allele MLM loss | Delta vs A0 | Epitope MLM loss | Delta vs A0 | MHC margin |
|---|---|---|---|---|---|
| A0 baseline | 2.257 | n/a | 2.225 | n/a | +0.751 |
| A1 embedding | 2.215 | -0.042 | 2.234 | +0.009 | +0.749 |
| A2 reset | 2.587 | +0.330 | 2.587 | +0.362 | +0.414 |
| A3 offset 50 | 2.213 | -0.045 | 2.301 | +0.075 | +0.570 |
| A4 full | 2.219 | -0.039 | 2.294 | +0.068 | +0.054 |
| A5 A4 plus augmentation | 2.271 | +0.013 | 2.294 | +0.068 | -0.149 |

Standard errors on the deltas are 0.005 to 0.014 nats, so the loss differences are resolved. All six
arms completed 300 steps at 650M. Three things follow.

**The module's cost is in the MHC margin, and it scales with how much of the module is switched on.**
Against A0's +0.751: the chain embedding alone is free (+0.749), the offset alone costs 0.18, the
per-chain reset alone costs 0.34, all three together cost 0.70, and adding order augmentation drives
the margin negative at -0.149. A negative margin means the scorer does worse than a uniform prior
over the 20 amino acids on the residues the project cares about. Held-out MLM loss does not show
this: A4 is 0.039 nats *better* than baseline on held-out alleles while its margin collapses. A
reader checking only the headline loss would conclude the full module is free.

**Per-chain reset is the worst single component**, at both sizes. It costs 0.33 nats on held-out
alleles, 0.36 on held-out epitopes, and raises the raw length-sensitivity slope to 0.0120 against
A0's 0.0037. It is the component that discards incoming position information, so a penalty is
expected; its size is the finding.

**Gate G2 passed on A4 at 650M, and the arm ran to completion.** At 35M the same arm was killed at
step 30, on a held-out MHC margin of -0.26 and a length sensitivity of 0.0155 against A0's 0.0034.
That kill was a property of the 35M backbone, not of the module. It is kept in the record as a
correctly-fired gate on the evidence available at the time, and withdrawn as a statement about the
module. The 650M margin of +0.054 says the module nearly exhausts the scorer's margin rather than
destroying it.

The 35M arms, for comparison:

| Arm | Allele MLM loss | Delta vs A0 | Epitope MLM loss | MHC margin |
|---|---|---|---|---|
| A0 | 2.443 | n/a | 2.451 | -0.024 |
| A1 embedding | 2.345 | -0.098 | 2.347 | -0.091 |
| A2 reset | 2.608 | +0.166 | 2.622 | -0.071 |
| A3 offset 50 | 2.329 | -0.113 | 2.361 | -0.119 |
| A4 full | killed at G2 step 30 | n/a | n/a | n/a |
| A5 A4 plus augmentation | 2.295 | -0.148 | 2.439 | -0.097 |

Every 35M arm including A0 ends with the MHC margin at or below zero. The report previously read that
as "LoRA at this scale damages the scorer regardless of the module". The 650M baseline at +0.751
shows the first half of that was about the scale: LoRA does not inherently damage the scorer, a 35M
backbone has no margin to spare. The conclusion that the module costs margin survives, and at 650M it
is measurable rather than buried under a floor effect.

Length sensitivity does not separate the arms at either size, because the shuffled null tracks the
observed slope in each. Excess over the null is between -0.00044 and +0.00007 for all six 650M arms,
which reproduces 8.4 under training.

### 8.7 The binding classifier: H2 tested at both sizes

H2 asks whether the chain-aware module improves held-out binding prediction. The masked-LM arms in
8.6 cannot answer it, so a binary classifier was trained on the same arms: LayerNorm plus Linear on
the mean over residue tokens, 2,400 positives from the `train` shard plus one decoy each, 300 steps
at batch 16, three seeds per arm, at ESM-2 35M and 650M. Mean pooling rather than `<cls>`, which
ESM-2 never trains as a sequence summary. Trainable parameters are matched within each size:
1,663,201 at 35M and 12,176,641 at 650M. Evaluation is fixed across seeds, arms and sizes:
`test_allele` is 1,000 positives plus decoys over 7 alleles, `test_epitope` is 600 positives plus
decoys over 5. Decoys are built inside each held-out shard, so decoy antigens are held out too. A4
trains here without hitting a gate, because G2 is defined on the masked-LM objective.

The learning rate was picked on A0 at 35M by held-out-allele AUROC across 1e-4, 5e-4 and 1e-3, which
gave 0.590, 0.628 and 0.600. There was no validation eval, so this is mild test-set selection applied
only to the baseline, which if anything favors A0. 5e-4 was then used for every arm at both sizes.

| Size | Arm | Allele AUROC mean (sd) | Epitope AUROC mean (sd) |
|---|---|---|---|
| 35M | A0 | 0.537 (0.097) | 0.466 (0.036) |
| 35M | A4 full | 0.476 (0.120) | 0.570 (0.043) |
| 35M | A5 A4 plus augmentation | 0.581 (0.062) | 0.547 (0.026) |
| 650M | A0 | 0.595 (0.064) | 0.459 (0.030) |
| 650M | A4 full | 0.468 (0.111) | 0.526 (0.024) |
| 650M | A5 A4 plus augmentation | 0.635 (0.019) | 0.559 (0.031) |

**No arm is an above-chance binding predictor** (`fig7_h2_classifier.png`). Every AUROC interval covers 0.5 except A0 on
held-out epitopes, which sits *below* chance at both sizes. On the question H2 actually asks, whether
the module buys usable binding performance, the answer is no at both sizes.

**The arm differences are not noise, which the arm means hide.** Seeds are shared across arms: seed
k fixes the training subsample, the shared initialization and the batch order, so A0 seed k and A4
seed k differ only by the module. Comparing arm means throws that pairing away and buries the shift
under between-seed variance several times larger. Paired by seed
(`fig8_h2_paired.png`, `src/cla/eval/h2_paired.py`, `results/phase2/h2_paired/`):

| Arm | Shard | 35M delta | 650M delta | Seeds agreeing |
|---|---|---|---|---|
| A4 | held-out allele | -0.061 (p 0.044) | -0.127 (p 0.099) | 6 of 6 negative |
| A4 | held-out epitope | +0.104 (p 0.147) | +0.067 (p 0.014) | 6 of 6 positive |
| A5 | held-out epitope | +0.082 (p 0.114) | +0.100 (p 0.009) | 6 of 6 positive |
| A5 | held-out allele | +0.044 (p 0.646) | +0.040 (p 0.277) | 3 of 6 positive |

Three contrasts point the same way in every seed at both model sizes. The module raises held-out
*epitope* AUROC and the full module lowers held-out *allele* AUROC, and both replicate across a
19-fold change in backbone size. The direction is what the project predicts: a baseline that leans on
chain-layout geometry should generalize worst to antigens it has never seen, and removing the
geometry cue should help there most.

**How far this goes.** Not far, for four reasons stated together because any one of them alone would
overstate the result.

1. The baseline is below chance on held-out epitopes, so the module moves it toward 0.5 rather than
   above it. The best arm, A5 at 650M, reaches 0.559 with a clustered interval of [0.437, 0.568].
   Making a model less anti-predictive is not the same as making it predictive.
2. Three seeds per cell. A sign test across the six seeds of a replicated contrast gives p = 0.031
   two-sided; correcting for the four contrasts examined leaves nothing significant. The
   replication across sizes is the evidence, and it is suggestive rather than conclusive.
3. The paired analysis was chosen after seeing the unpaired result. It is the right analysis for a
   shared-seed design and would have been the right one to pre-register, but it was not
   pre-registered, and an analysis picked post hoc is worth less than one fixed in advance.
4. Negatives are mismatched decoys, a real receptor paired with the antigen of a different epitope.
   Some are false negatives from cross-reactivity, and unrelated antigens are easier to separate
   than genuine non-binders. The notice is carried in `src/cla/train/classifier.py` and in every
   metrics JSON it writes.

**Verdict on H2: not supported, with a consistent directional signal that a properly powered
experiment should chase.** The module does not deliver the held-out binding gain H2 claims. It does
move held-out-epitope AUROC up in all twelve arm-seed comparisons that test it, which is the
generalization axis the chain-layout argument predicts, and that pattern survived a change of model
size. The next experiment is more seeds and validated negatives on the epitope shard, not more arms.

One mechanism observation, kept as an observation. At 35M, A0 and A4 drive training loss from 0.69 to
0.16 to 0.32 while held-out AUROC stays near chance, the signature of fitting something that does not
transfer. A5, whose only difference from A4 is chain-order augmentation, never fits at 35M: final
training loss 0.682, 0.691 and 0.485. It also has the highest held-out epitope mean at both sizes.
That the augmentation removes a layout shortcut the other arms learn is consistent with these numbers
and is not established by them.

### 8.8 Phase 2 verdict

The module does what it was designed to do on the permutation axis and costs score quality to do it.
Trained directly on binding at two model sizes, it does not produce an above-chance predictor on
either held-out shard, so H2 as written is not supported. On the masked-LM objective the arms are
indistinguishable from baseline on held-out loss, but the module's cost shows up in the MHC margin
and scales with how much of it is enabled: from +0.751 at baseline to +0.054 for the full module and
-0.149 with order augmentation.

Against that, one pattern survived a 19-fold change in backbone size with every seed agreeing: the
module raises held-out-epitope AUROC and the full module lowers held-out-allele AUROC. That split is
what the chain-layout argument predicts, since a baseline leaning on layout geometry should fail
worst on antigens it has never seen. It is a direction, not a result: three seeds per cell, a
baseline below chance on the shard that improves, and an analysis chosen after seeing the data.

On the axis that motivated the project, native chain-length variation, there is no baseline effect
for the module to fix. CDR3-beta length moves offsets by 3 to 4 residues and the artifact needs 25.

The recommendation is against adopting the module for this setting, and in favor of one more
experiment before the question is closed: the held-out-epitope shard, with validated non-binders and
enough seeds to resolve a 0.05 to 0.10 AUROC shift.

What would change the recommendation: full-length TCR variable domains instead of CDR3 loops, where
length varies by far more than 3 to 4 residues and could reach the offset regime where Phase 1 shows
the model is genuinely sensitive; and a backbone fine-tuned for binding, where the score carries
signal worth protecting.

## 9. Conclusion

Where the estimator passes its own check, ESM-2 treats a content-free change in position indices at
chain breaks as 4.8 to 10.5 times as large as twenty real substitutions. The effect holds across 51
alleles at both 150M and 650M and acts on genuine cross-chain computation. Scale lowers it without
removing it. H1 is supported for the one target where it can be tested.

H2 is not. Trained on binding at two model sizes and three seeds each, no arm is an above-chance
predictor on either held-out shard. The deeper reason is that the artifact needs an offset of 25 or
more to appear, and the nuisance that motivated the project, CDR3-beta length variation, moves
offsets by 3 to 4. Phase 2 measured no length-attributable effect in the baseline, so the module has
nothing to fix on that axis, and the masked-LM arms that most reduce permutation sensitivity have
degraded into unusable scorers. The recommendation is against adopting a chain-aware positional
module for this setting.

One finding points the other way and is kept because it replicated. Paired by seed, the module
raises held-out-epitope AUROC in every seed at both 35M and 650M, and the full module lowers
held-out-allele AUROC in every seed at both sizes. That is the generalization split the chain-layout
argument predicts. It rests on three seeds per cell, a baseline below chance on the improving shard,
and a pairing chosen after seeing the data, so it is a direction for the next experiment and not a
result this one establishes.

The two results together are the useful output: a real positional artifact, and the finding that
natural chain geometry does not reach it. Either one alone would have been misleading. Reporting only
the first would have motivated a fix nobody needs; reporting only the second would have missed a
genuine defect that full-length chains could expose.

Phase 2 ran at dev-loop scale and is reported in section 8. What a funded Phase 2 would still need
to establish:

1. That a model fine-tuned for binding shows the same sensitivity.
2. Whether the replicated held-out-epitope gain is real, at enough seeds to resolve 0.05 AUROC and
   with validated non-binders, and whether the matching held-out-allele cost is the price of it.
3. A binding-relevant target that passes admissibility, through a trained scorer or an
   admissibility reference defined before the run that suits hypervariable spans.
4. Validated non-binders in place of mismatched decoys, and allele-balanced evaluation at every size.
5. The H3 test of whether gains concentrate where chain lengths deviate from the training mode.

Artifacts: `results/phase1/` (run `nsi.json` and `summary.tsv`, `mhc_cluster_ci.tsv`,
`context_gain.json`, `gate_g1_gsi.json`, `figures/`), `results/phase2/` (`zeroshot_gsi.json`,
`noncircular_eval.json`, `training/`, `classifier/`), `src/cla/model/positional.py`, `docs/DECISIONS.md`,
`docs/COMPUTE_LOG.md`, `data/SOURCES.md`.
