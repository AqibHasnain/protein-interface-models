# Decision log

Append-only. Newest last. Each entry: what was decided, the alternatives, why.

## 2026-10-10 - Compute runs locally on Apple Silicon, not in the cloud

Phase 1 is inference-only over at most a few thousand complexes at three model
sizes. An M4 Max with 128 GB of unified memory runs ESM-2 650M in float32
comfortably through PyTorch's MPS backend.

Alternatives: Kaggle (30 GPU-hr/week free), Colab, Lightning, Modal. All were
rejected for Phase 1 because they require account provisioning the agent cannot
do, and they add a data-transfer and session-timeout failure mode for a workload
that fits locally. They remain the route for Phase 2 training.

Consequence: no spend, no credentials, and the audit reproduces on any machine
with 16 GB of RAM at the 35M and 150M sizes.

## 2026-10-10 - ChainSpec separates the complex from its serialization

`ChainSpec` holds an ordered tuple of chains but exposes `identity`, the
multiset of (role, sequence) pairs. The null-relayout operators are then
definable as transformations that provably preserve `identity`, and the unit
tests assert exactly that. Had serialization been baked into the data structure,
"this perturbation is biologically null" would have been an argument in prose
rather than an assertion in code.

## 2026-10-10 - Splits are membership tests, not a random partition

Held-out alleles and epitopes live in `configs/splits.yaml` and an example is
assigned by testing membership, with a hash of the example id breaking out the
validation fraction. Allele names are normalized to two IMGT fields first, so
`HLA-A*02:01:01:01` cannot enter training while `HLA-A*02:01` is held out.

A held-out epitope also removes every TCR observed against it anywhere in the
data. Without that propagation a receptor could train on one annotation and be
evaluated on another, which is leakage that a random split would not catch.

## 2026-10-10 - TCR chains carry CDR3 sequences only

VDJdb gives CDR3 alpha and beta plus V/J gene calls. Reconstructing full-length
variable domains from germline would add an IMGT gene-sequence dependency and a
recombination model, both of which are their own source of error.

The audit's claim is about relative-position geometry across chain boundaries,
which CDR3-length variation already exercises: that is precisely the quantity
the plan's motivating example varies. Recorded as a limitation in the report.

Consequence: operator P3 (restore natively truncated flanking residues) has no
flank data to work with and reports as not-applicable rather than being faked
with invented residues.

## 2026-10-10 - Token dropout is a confound the audit must control for

HuggingFace ESM-2 sets `token_dropout=True`: masked positions are zeroed and
every embedding in the sequence is rescaled by the observed mask fraction
(`modeling_esm.py` lines 241-248). Span pseudo-log-likelihood masks a whole
chain at once, so the mask fraction is (masked span length) / (total length).

Operators P2 and P4 change the total length. They therefore change that
fraction, and so move the score through a path that has nothing to do with
relative-position geometry. Reporting a single NSI without separating the two
would attribute a normalization artifact to the positional encoding, which is
the exact error this project exists to avoid making about other people's models.

Decision: every operator is scored twice, with `token_dropout` on and off. The
headline NSI uses the model as published (dropout on), because that is what a
user of the model actually experiences. The dropout-off arm is reported beside
it and is what licenses the attribution to position geometry. P1 changes no
length, so agreement between its two arms is the internal check that the
control is working.

Alternative considered: always disable token dropout. Rejected because it
measures a model nobody runs.

## 2026-10-10 - ESM-2 positions enter only through rotary embeddings

`position_embedding_type="rotary"`; `EsmEmbeddings` adds absolute position
embeddings only in the `"absolute"` branch, which ESM-2 never takes.
`EsmModel.forward` passes `position_ids` into `EsmRotaryEmbedding`, defaulting
to `arange(seq_len)`, and every `EsmSelfAttention` rotates queries and keys with
the result.

Two consequences. Rotary attention depends only on position differences, so a
constant shift of all ids is a no-op and only cross-chain offsets bite, which is
exactly the quantity H1 is about. And the Phase 2 module intervenes by supplying
`position_ids`, with no monkeypatching of the backbone: that supported argument
is what makes the fix genuinely drop-in.

## 2026-10-10 - MHC trimmed to the alpha-1/alpha-2 groove; complexes deduplicated on identity

The MHC heavy chain is cut to residues 1-180 of the mature chain, selectable through
`mhc_domains` (`a1a2`, `ectodomain`, `full`) and recorded in the output file. Only
the groove contacts the peptide and the TCR, and it is the usual truncation in pMHC sequence
models. The full chain left too little headroom under 512 tokens for P2 and P4 to add tokens,
and an audit whose perturbations silently truncate the MHC would measure truncation rather
than position geometry.

Alternative: keep the full chain and accept truncation or a longer limit. Rejected because the
512-token budget is fixed for the audit, and the extra ~160 residues carry no peptide or TCR
contacts.

Specs are deduplicated on `ChainSpec.identity`, keeping the highest `vdjdb.score` in each
group, before saving. Repeated complexes would weight every median toward whatever is
replicated and violate the independence the bootstrap assumes, which would make the reported
confidence intervals too narrow.

Alternative: deduplicate downstream in the audit. Rejected because every consumer of the file
would then have to remember to do it.

## 2026-10-10 - The pre-registered NSI denominator is unresolvable on an untrained backbone

A 60-example pilot at 35M showed the median span-pLL gap between VDJdb binders
and mismatched decoys is about 0.0004 nats with token dropout on, and 0.019 with
it off. NSI divides by that gap, so it came out between 90 and 860 with
bootstrap intervals that do not bracket the point estimate. Those numbers are
arithmetic noise, not evidence of layout sensitivity.

The cause is not a bug. Zero-shot ESM-2 is not a binding predictor. It was never
trained to score whether a receptor binds a peptide, and the plan's Phase 2 is
what turns it into one. Phase 1 nonetheless has to audit the untrained baseline,
so it needs a signal scale that exists before any training.

**Decision.** Report two denominators, both labelled.

1. The pre-registered NSI stays in the output and keeps its pre-registered gate
   thresholds. Where the denominator is unresolved it is reported as not
   computable, with the measured separation attached, rather than as a large
   number. A metric that cannot be computed is not a result.

2. **NSI-C**, the primary reportable for Phase 1, normalizes by the positive
   control instead of by the binder/non-binder gap:

   ```
   NSI-C = median |sPLL(x) - sPLL(null_relayout(x))| / median |sPLL(x) - sPLL(P5(x))|
   ```

   P5 swaps in a genuinely different CDR3-beta of different length, so the
   denominator is the score movement caused by a real biological change. NSI-C
   asks directly: does writing the same complex down differently move the score
   as much as making it a different complex? That is H1 restated without
   reference to any trained discriminator, and P5 was already built to be
   exactly this quantity. NSI-C >= 1.0 means the artifact is as large as the
   biology.

**Disclosure.** The NSI-C threshold of 0.25, carried over unchanged in form from
the pre-registered NSI threshold, was fixed after the 60-example pilot had been
seen. The pilot measured the denominator, not NSI-C, and the threshold was not
tuned to produce any particular verdict, but it is not a blind pre-registration
and the report must say so in those words.

Alternative considered: obtain real validated non-binders from the Messemaker
set and keep the original denominator. Rejected for Phase 1 because real
negatives do not fix the actual problem. An untrained ESM-2 does not separate
binders from non-binders regardless of how good the non-binders are, and the
pilot's near-zero gap is a property of the model, not of the decoys.

## 2026-10-10 - Zero-shot ESM-2 does not discriminate TCR-pMHC binding

Recorded as a finding in its own right rather than as an obstacle. The span
pseudo-log-likelihood of a peptide conditioned on a concatenated TCR and MHC
context is near-identical for observed pairs and for mismatched decoys. Anyone
building on a concatenation-based ESM-2 baseline without fine-tuning should know
this before interpreting a score from it. It belongs in the audit write-up.

## 2026-10-10 - Separator tokens are content, so the audit is redesigned around pure-geometry operators

An adversarial review of the Phase 1 code found that the perturbations did not
isolate what they claimed. Three confirmed defects, measured at 35M on CPU:

1. The separator was `<pad>` with `attention_mask=1`. Its embedding row has norm
   5.83, so k separators are k out-of-distribution content tokens that every
   layer attends to. Decomposing the P2 shift at 25 separators into a tokens arm
   and a positions-only arm put over 99% of it in the tokens arm. P2 and P4 were
   measuring token insertion, not relative-position geometry.
2. The P5 positive control moved scores less than the nulls it was meant to
   bound (median 0.041 nats against P4's 0.37). Zero-shot ESM-2 largely ignores
   the receptor, so swapping CDR3-beta changes the score mainly through the same
   length and position path the nulls use. NSI-C was a ratio of one nuisance to
   another and could be pushed above 1.0 by picking a bigger null.
3. P1 left the first chain adjacent to `<cls>` and the last to `<eos>`. A
   peptide in the first slot moved 2.5x to 6x more than one in the middle, so
   part of P1 was special-token proximity rather than inter-chain distance.

Separately, P2 drew its own baseline value (1 separator) on 25% of draws,
rendering an identical string and contributing exact zeros to the median.

**The redesign.** ESM-2 honors `position_ids` and positions enter only through
rotary embeddings. That permits a perturbation with no content change at all:
hold `input_ids` byte-identical and change only `position_ids`. Nullness stops
being an argument about biology and becomes tensor equality, which a test can
assert directly. These are the G operators, and they carry the headline.

- **G1** adds a constant index gap at every chain break.
- **G2** adds a gap before the peptide only.

Rotary attention depends on index differences, so these change exactly the
cross-chain relative offsets that H1 is about, and nothing else. Baseline
rendering now uses zero separator tokens, leaving the position gap as the only
chain-boundary signal.

The content-coupled operators are kept and reported separately under honest
labels, because "what happens if you insert linker tokens" is a real question,
just not a question about geometry. P1 now renders with separators on both ends
so no chain ever touches a special token, and P2 no longer draws its baseline.

**Positive control.** Normalizing by a control that moves less than the nulls is
worthless. The new control mutates a matched number of residues in a context
chain, so the denominator is the score movement caused by a real sequence change
of comparable size. Reported alongside the old CDR3-beta swap.

**Scoring target.** The peptide span sits near the model's prior: median span
pLL is -3.04 against ln(20) = 3.00 for a uniform prediction. Shifts of 0.05 nats
on an essentially flat landscape cannot carry a headline. MHC_HEAVY is added as
a target because it is a conserved protein the model predicts well, giving the
measurement real dynamic range. All three targets are reported.

Cost of this decision: the pre-registered operator set P1 to P5 is no longer the
primary analysis. The report must present the original set, the defects found,
and the redesign, rather than quietly shipping the new operators as if they had
been the plan. The pre-registered gate thresholds are not changed.

## 2026-10-10 - Confirmed non-defects, recorded so they are not re-investigated

Batch composition and right-padding do not affect scores: token dropout's
rescale divides by `attention_mask.sum(-1)`, not the padded width, and batches
of 1, 16 and reversed order agree to 9.5e-7. Operator applicability exclusions
stay index-aligned between numerator and denominator, and the bootstrap
resamples every array on shared indices. `chain_slices` follows a chain to its
new slot after permutation, and the scored span length is invariant across the
null operators.

## 2026-10-10 - The premise survives: geometry moves scores more than real mutations do

A 60-example probe at 35M, median rendered length 319, measuring median absolute
change in span score against a gap-0 baseline:

| Perturbation | PEPTIDE | TCRB | MHC_HEAVY |
|---|---|---|---|
| G1 gap 0 (sanity) | 0.000 | 0.000 | 0.000 |
| G1 gap 5 | 0.129 | 0.033 | 0.460 |
| G1 gap 25 | 0.097 | 0.036 | 1.435 |
| G2 gap 25 | 0.084 | 0.035 | 0.153 |
| C1 1 mutation | 0.011 | 0.002 | 0.039 |
| C1 5 mutations | 0.039 | 0.010 | 0.149 |
| C1 20 mutations | 0.084 | 0.018 | 0.560 |

A position-index gap of 5, inserted with `input_ids` bitwise unchanged, moves
the peptide score further than twenty real amino-acid substitutions. The gap-0
row is exactly zero, which is the sanity check that the operator is inert when
it should be.

This is H1 restated in the cleanest available form, and it is the Phase 1
headline. Gate G1 is not STOP.

Three constraints the full audit must respect.

**Gaps stay in distribution.** ESM-2's trained position range is 1026. A gap of
200 across four chain breaks pushes indices to about 1124, so the gap-200
numbers measure extrapolation past the trained range rather than relative
offset. Gaps are capped at 150, and anything beyond is reported separately and
labelled out of distribution.

**MHC_HEAVY is diagnostic, not headline.** Its span pLL is -5.619 against a
uniform-prior reference of -3.00, so masking all 180 residues at once leaves the
model predicting worse than chance. Its large apparent gap sensitivity is partly
a measure of how badly the estimator degrades on a fully masked long span.
PEPTIDE (-3.119) and TCRB (-2.916) sit near the prior and carry the headline.

**The P1 baseline has to match.** `symmetric_ends` renders a separator run of
length `separator_tokens`, which is now 0 by default, so it removes the
special-token adjacency confound only when at least one separator is rendered.
P1 and its baseline are both rendered with one separator and symmetric ends, or
the confound returns.

## 2026-10-10 - Geometry sensitivity is graded for some target/operator pairs and a step for others

Full sweep at 35M, n=1000, paired bootstrap CIs, token dropout on. GSI is the
median absolute score shift divided by the shift from twenty real amino-acid
substitutions in a context chain.

G1 inserts the gap at every chain break; G2 inserts it only before the peptide.

| Gap | PEPTIDE G1 | PEPTIDE G2 | TCRB G1 | MHC_HEAVY G1 |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 1.26 | 0.60 | 1.21 | 0.42 |
| 2 | 1.35 | 0.94 | 1.17 | 0.57 |
| 5 | 1.33 | 1.26 | 1.50 | 1.00 |
| 10 | 1.09 | 1.63 | 1.45 | 1.20 |
| 25 | 1.16 | 1.91 | 2.09 | 1.73 |
| 50 | 1.20 | 2.04 | 2.46 | 2.16 |
| 100 | 1.18 | 2.76 | 2.01 | 2.81 |
| 150 | 1.17 | 3.32 | 3.64 | 3.63 |

Gap 0 is exactly zero everywhere, which is the inertness check on the operator.

**The answer is both, depending on where the gap goes.** A gap at every chain
break saturates immediately for the peptide target: one index is worth as much
as 150. A gap placed only before the peptide is graded and monotonic, reaching
3.32 at gap 150. MHC_HEAVY under G1 is cleanly graded across the whole sweep.

This matters for Phase 2. For the peptide target the chain-break offset is a
step function, so the structure-model convention of a large constant (200) buys
nothing over a small one, and the offset magnitude is not worth tuning. The
graded cases say the opposite, so the ablation has to report the offset value as
a swept parameter rather than a fixed 200.

**The caveat that must travel with every one of these numbers.** The
denominators are small: twenty real substitutions move the peptide score 0.043
nats, TCRB 0.018, MHC_HEAVY 0.014. GSI exceeds 1.0 partly because zero-shot
ESM-2 barely responds to real sequence change either. The honest claim is that
layout noise is comparable to or larger than the biological signal in a model
whose score is weakly driven by both, not that layout noise is large in
absolute terms. Any version of this result that quotes GSI without the
denominator is misleading, so the report prints raw nats beside every ratio.

## 2026-10-10 - Correction: the step finding does not replicate, and the artifact grows with scale

The 150M run contradicts two things recorded from the 35M run. Both corrections
stand; the earlier entry is left in place because this log is append-only.

**The peptide G1 step is a 35M artifact.** At 35M, GSI was flat from gap 1
onward (1.26, 1.35, 1.33, 1.09, 1.16, 1.20, 1.18, 1.17). At 150M the same curve
rises and then falls (0.79, 1.37, 1.81, 2.08, 2.50, 2.43, 2.09, 1.43), peaking
near gap 25. The earlier conclusion that the chain-break offset is a step
function, and therefore that its magnitude is not worth tuning in Phase 2, is
withdrawn. The offset must be swept, not fixed at the structure-model
convention of 200.

**The artifact gets worse with model size.** MHC_HEAVY under G1 at gap 150 goes
from GSI 3.63 at 35M to 12.46 at 150M, and the gap-50 value from 2.16 to 11.52.
TCRB under G1 rises similarly. Nothing in the audit predicted the direction, and
the naive expectation would have been that a larger model handles layout better.

That direction is the most consequential thing Phase 1 has produced. If it holds
at 650M it means the omission does not wash out with scale, which is the
assumption that would otherwise justify ignoring it. It also means the 650M
number is load-bearing rather than a formality, so the reduced n=250 there is a
real limitation and the report must say that the strongest claim rests on the
smallest sample.

Two scale points do not establish a trend. The report states this as a direction
observed across two sizes with a third pending, not as scaling behaviour.

## 2026-10-10 - The estimator sanity check fails for two of three targets

The audit records, per target, the median baseline span pLL against the
uniform-prior reference of -2.9957 (-ln 20), with a required margin of 0.25.

| Target | Estimator | Margin | Usable |
|---|---|---|---|
| PEPTIDE | span_pll | 0.039 | no |
| TCRB | span_pll | 0.104 | no |
| MHC_HEAVY (35M) | windowed_span_pll | 0.068 | no |
| MHC_HEAVY (150M) | windowed_span_pll | 0.627 | yes |

Masking an entire short span removes all local context, so the model falls back
to roughly the uniform prior. A score that carries almost no information cannot
support a claim about how that score moves. This is the same defect that whole
span masking produced on MHC_HEAVY, where windowing fixed it, and the peptide
is a 9-mer so the problem is worse there, not better.

**Consequences, stated before any attempt to rescue the result.**

The PEPTIDE and TCRB numbers at both model sizes are not interpretable as they
stand. That is most of the audit.

The scale claim is withdrawn. The MHC rise from GSI 3.63 at 35M to 12.46 at
150M compares an estimator that fails the sanity check against one that passes
it, so it is not evidence about scale. The previous entry recording that rise as
the most consequential Phase 1 finding is superseded. Two of the three model
sizes contributing to it were never admissible.

One condition survives unchanged: MHC_HEAVY at 150M, margin 0.627, where a
content-free position-index change moves the score 0.55 nats, against 0.045
nats for twenty real substitutions.

The sanity threshold was fixed at 0.25 in the audit code before these numbers
were read, and it is not being moved now that it has excluded results. A
threshold that is relaxed when it rejects the desired answer is not a check.

## 2026-10-10 - No estimator rescues the short spans, and the threshold is not being moved

A probe at 35M over 200 specs tried span_pll, windowed masking at (1,1), (2,2),
(3,3), (5,5), and full pseudo-likelihood on PEPTIDE and TCRB. Nothing clears the
0.25 margin. The best TCRB margin is 0.139 from whole-span masking, and
shrinking the window makes it worse. PEPTIDE never rises above the uniform prior
under any estimator.

**The span-masking estimator was not the cause.** Three measurements say so.

A unigram model fitted to the dataset's own spans reaches a margin of 0.40 on
TCRB and 0.15 on unique epitopes. ESM-2 35M reaches 0.10 on TCRB. The model is
beaten by amino-acid frequencies on the receptor span.

Scoring each span alone and then inside the full complex gives the context gain,
which is how much the rest of the complex improves the prediction. Under
windowed estimators it is +0.04 to +0.07 for TCRB and -0.09 to -0.13 for
PEPTIDE. The model extracts no usable information from the other chains, and for
the peptide the other chains actively hurt.

MHC_HEAVY passes because it is conserved and predictable. CDR3 and epitope
residues are hypervariable, so a near-uniform masked-LM prior there is the
expected outcome rather than a defect.

**Decision: the 0.25 threshold stays where it is, and the uniform-prior
reference stays too.** Both are arguably miscalibrated for hypervariable spans,
and a dataset-unigram reference would be the better instrument. That change is
not being made now, because the GSI outcome has already been seen. Rewriting the
admissibility criterion after learning which results it excludes would convert a
check into a formality. It is recorded as the first item of future work, to be
fixed before any rerun and evaluated blind.

**Consequence for the report.** PEPTIDE and TCRB are reported as not
established. Their GSI exceeds 1.0 under every estimator tried, rising to 7x and
11.5x at gap 150, and gap 0 is exactly 0 throughout. That is consistent with the
hypothesis and is worth showing, but it is movement in a score that carries
almost no information, so it cannot carry a claim. MHC_HEAVY at 150M is the one
condition that passes its own admissibility check and it is what the headline
rests on.

## 2026-10-10 - Context gain is promoted to a first-class diagnostic

The quantity that settled the question above deserves to be reported rather than
buried in a probe: the difference between a span's pLL scored alone and scored
inside the complex. It answers directly whether a concatenation-based model uses
the other chains at all, which is a precondition for any claim about chain
layout mattering.

It also sharpens the finding. The score moves when chain-break geometry changes
while the context gain is approximately zero, so the sensitivity is not a
corruption of cross-chain reasoning the model is doing. There is no cross-chain
reasoning to corrupt. The position change is reaching the score through a path
that carries no information about the complex.

Anyone proposing to fix chain layout in a concatenation model should measure
context gain first. If it is zero, the layout fix addresses an artifact in a
model that is not using the chains anyway, and the honest move is to fix that
instead.

## 2026-10-10 - 650M admits a second usable point, and layout sensitivity falls with scale

The windowed MHC_HEAVY estimator by model, against the uniform prior of -2.996:
35M -2.937 (fails the 0.25 margin), 150M -2.392 (passes), 650M -1.831 (passes,
by the widest margin). Two admissible model sizes rather than one.

MHC_HEAVY under G1, token dropout on, at matched gaps:

| Gap | 150M GSI | 150M raw nats | 650M GSI | 650M raw nats |
|---|---|---|---|---|
| 1 | 0.71 | 0.032 | 0.84 | 0.030 |
| 5 | 2.28 | 0.101 | 1.85 | 0.066 |
| 25 | 8.59 | 0.383 | 4.83 | 0.172 |
| 100 | 10.52 | 0.469 | 5.08 | 0.181 |

**Layout sensitivity decreases with model size.** Across the two admissible
points it roughly halves, in the ratio and in absolute nats alike. This is the
opposite of the direction recorded earlier from 35M and 150M, which compared an
inadmissible estimator against an admissible one and has already been withdrawn.

The effect does not vanish. At 650M a content-free position-index change still
moves the score about five times as far as twenty real substitutions. The
defensible statement is that scale attenuates the artifact without removing it,
established on two points, which is a direction and not a scaling law.

## 2026-10-10 - The GSI gate reads PROCEED, but not for the reason the driver reports

`aggregate.py` returns PROCEED on PEPTIDE and TCRB across all three models.
Those two targets fail the estimator sanity check at every size, so a gate
resting on them rests on scores that carry almost no information.

The gate is nonetheless PROCEED, on the admissible evidence alone: MHC_HEAVY at
150M and 650M, where the estimator passes, GSI runs from 4.83 to 10.52 at gaps
25 and 100, and every CI lower bound sits far above the 0.5 requirement.

The report states the verdict together with the subset it rests on. A gate
passed on inadmissible targets would have been failed by its own criteria.

## 2026-10-10 - Correction: the model does use cross-chain content at the usable sizes

The earlier entry promoting context gain concluded there is no cross-chain
reasoning for a layout change to corrupt. That was measured only at 35M, where
the estimator fails its sanity check, and it does not hold where the estimator
passes.

A scrambled-context control settles it. Each other chain's residues are shuffled,
preserving length and composition but destroying content. Genuine content use
requires complex > scrambled, not merely complex > alone.

MHC_HEAVY, median nats:

| Model | Arm | complex - alone | complex - scrambled |
|---|---|---|---|
| 35M | on | +0.070 | +0.028 |
| 35M | off | +0.657 | -0.093 |
| 150M | on | +0.647 | +0.599 |
| 150M | off | +0.661 | +0.612 |
| 650M | on | -0.032 | +0.190 |
| 650M | off | +0.542 | +0.216 |

At 150M and 650M the content term is clearly positive and stable across both
token-dropout arms, so it is not the mask-fraction rescaling artifact. The model
extracts real, content-dependent information from the other chains for the MHC
span: about 0.6 nats at 150M and 0.2 at 650M.

At 35M the content term is approximately zero and flips sign between arms, which
is consistent with the estimator there being unusable.

**The headline becomes the stronger claim rather than the weaker one.** Where
the model demonstrably integrates across chains, a position-index change that
alters no content disturbs the score more than twenty real substitutions do.
That is H1 as originally stated, now resting on a condition where there is
genuine cross-chain computation to disturb.

The `complex - alone` comparison is confounded at 650M, where it reads -0.03
with dropout on and +0.54 with it off, because the lone chain and the full
complex have different mask fractions. The scrambled control is the measure to
cite; `complex - alone` is reported beside it but is not the evidence.

## 2026-10-10 - The MHC confidence intervals are pseudo-replicated and must be recomputed

A 300-example draw contains only 11 distinct MHC heavy-chain sequences. VDJdb is
dominated by a few common alleles, HLA-A*02:01 above all. The bootstrap
resamples examples and therefore treats repeated alleles as independent
observations, which makes every MHC interval far too narrow. The context-gain
CIs of about +/-0.002 are not credible on 11 underlying sequences.

This applies to the headline. The MHC_HEAVY GSI intervals in the main audit come
from the same example-level bootstrap over a sample with the same allele
concentration.

The correct unit of resampling is the distinct MHC allele, not the example. The
MHC numbers are recomputed with a cluster bootstrap over alleles, and the
example-level intervals are not reported. Widening is expected and is the
honest width.

Recorded before seeing the recomputed intervals, so that whatever they come out
as is what gets published.

## 2026-10-10 - The headline survives the cluster bootstrap, but the sample is two alleles deep

Recomputing the MHC_HEAVY intervals with the allele as the resampling unit
widens them by 1.6x to 3.8x, as expected, and the headline holds.

| Model | Gap | GSI | Cluster CI | Example CI |
|---|---|---|---|---|
| 150M | 25 | 8.59 | [6.77, 9.40] | [7.89, 9.51] |
| 150M | 100 | 10.52 | [9.78, 16.11] | [9.67, 11.64] |
| 650M | 25 | 4.83 | [1.96, 5.66] | [4.24, 5.77] |
| 650M | 100 | 5.08 | [2.39, 5.97] | [4.42, 6.04] |

Gaps 1 and 2 now fall below 1.0 under the correct interval, where the
example-level bootstrap had them marginally above. Those small-gap claims are
withdrawn. Gap 0 remains exactly 0.0.

**The diagnostic matters more than the intervals.** The audited sample holds 14
distinct alleles at 150M and 9 at 650M, HLA-A*02:01 accounts for 69% of
examples, and Kish's effective sample size is **1.92**. The cluster bootstrap is
the right estimator, but run over an effective two alleles it cannot establish
that the effect generalizes across MHC alleles. The result could be a property
of HLA-A*02:01 and this data could not tell the difference.

This is now the single largest threat to the Phase 1 claim, ahead of the
estimator question. The example count of 1000 was never the real sample size.

An allele-balanced confirmatory run is in progress: equal quotas per allele,
MHC_HEAVY and G1 only, at 150M and 650M, reporting per-allele GSI so
heterogeneity is visible rather than averaged away. The report states the
headline at whatever strength that run supports, and reports the per-allele
spread regardless of which way it comes out.

## 2026-10-10 - The headline generalizes across MHC alleles

Allele-balanced resampling (51 distinct alleles, capped at 27 examples each,
n=610) raises Kish's effective sample size from 1.92 to 26.86. MHC_HEAVY, G1,
150M, token dropout on:

| Gap | Balanced GSI | Concentrated GSI |
|---|---|---|
| 0 | 0.00 | 0.00 |
| 1 | 0.50 [0.43, 0.59] | 0.71 [0.68, 0.92] |
| 5 | 1.79 [1.58, 2.04] | 2.28 [1.81, 2.96] |
| 25 | 6.96 [6.18, 7.85] | 8.59 [6.77, 9.40] |
| 100 | 11.71 [10.39, 13.19] | 10.52 [9.78, 16.11] |

The two samples agree. The effect is not a property of HLA-A*02:01, which had
been the live alternative explanation while the effective sample was two
alleles. Gap 0 is exactly 0.0, and gap 1 stays below 1.0, consistent with the
withdrawal of the small-gap claims.

This is the last threat to the Phase 1 headline that the available data can
address. The 650M balanced run is still in progress and will be added when it
completes; the 150M result is what the claim rests on.

## 2026-10-10 - Phase 2 scope under local-only compute

Gate G1 passed, so the plan's Phase 2 runs. The plan budgets up to $300 on
A100-class GPUs for six LoRA arms at 650M. No cloud account exists and the
budget is zero, so Phase 2 is executed at the plan's own dev-loop scale
(section 7.3: 150M, subsampled data, free tiers) on local Apple Silicon.

One property of the backbone makes most of H2 testable for nothing. ESM-2
positions enter only through rotary embeddings and the model accepts
`position_ids`, so two of the three module components, per-chain index reset and
chain-break offset, are pure index remaps with no parameters. Arms A2, A3 and
their combination can therefore be measured zero-shot on the existing audit
machinery. Only the chain-identity embedding (A1) needs gradients, and A4 and A5
need training to combine all three.

Consequence for the claim. The zero-shot arms answer whether the module reduces
the measured artifact. They do not answer whether it helps held-out binding
generalization, because the backbone cannot discriminate binding at all, which
Phase 1 established. The H2 generalization claim is out of reach at this budget
and is reported as not tested rather than as a null result.

The offset magnitude is swept rather than fixed at the structure-model
convention of 200, because the earlier claim that magnitude does not matter was
withdrawn when the 35M step turned out not to replicate.

## 2026-10-10 - CI runs green locally; coverage is below the plan's target

The workflow's steps (ruff, the leakage test, the full suite) pass on this
machine. Line coverage on `src/cla` is 58% against the plan's 80% target.

The shortfall is concentrated in driver modules: the audit, figure, aggregate
and allele-balanced entry points. Those are exercised end to end by the real
runs whose outputs are committed in `results/`, rather than by unit tests.

Coverage is not raised by adding tests to hit those lines. The repository's
testing rule treats a test that exists to move a coverage number as bloat, and
the plan's 80% figure is a target rather than a finding. The gap is recorded
here instead of being closed with tests that guard nothing.

## 2026-10-10 - The reset arm's GSI of zero is circular and is not reported as a result

Measuring the per-chain-reset arm against the G1 operator is a tautology. G1
perturbs by adding a gap to position indices at chain breaks. Per-chain reset
rebuilds every index from `chain_slices` and discards whatever came in. The arm
therefore scores exactly 0 by construction, for the same reason that deleting a
number after adding it returns the original.

The zero is an implementation check that reset does what it says. It is not
evidence that the module fixes anything, and the Phase 2 results present it as
such explicitly rather than as A2 winning.

**What the non-circular test has to be.** H1's motivating nuisance is not an
injected gap, it is variable chain length: a CDR3-beta three residues longer
shifts every downstream cross-chain offset by three. The fix is supposed to make
the score insensitive to that. Two operators test it without circularity:

- **Chain order permutation.** Permuting chains genuinely changes the token
  sequence, so the module cannot erase it. Under the baseline the permutation
  also changes every cross-chain offset; under reset plus a constant break
  offset the within-chain indices and the chain-to-chain separation are both
  order-invariant. Residual sensitivity from attention over reordered tokens
  remains in both arms, so the comparison is informative rather than definitional.
- **Native chain-length variation.** Complexes sharing an allele but differing in
  CDR3 length, measuring the variance of the MHC score across them. Under the
  baseline, length drives downstream offsets; under the fix it should not.

Both are run before any Phase 2 claim is made.

## 2026-10-10 - The module is not free at inference, and offset 200 does not fit ESM-2

Two practical findings from the zero-shot arms, both worth more than the GSI
numbers around them.

**Applying the module degrades the zero-shot score.** Median baseline score
change against A0, with the resulting margin over the uniform prior where
measured: reset alone moves the score 0.66 nats and leaves a margin of 0.03,
offset 50 moves it 0.47 for a margin of 0.22, and reset plus offset 200 moves it
0.42 for 0.28. The Phase 1 usability threshold is 0.25, so several arms land at
or below the bar they would need to be measurable at all.

This is expected and it is the reason the plan trains the arms rather than
applying the module zero-shot. ESM-2 has never seen per-chain-reset positions.
It does mean the module cannot be sold as a free inference-time fix, which is
how a reader would otherwise interpret a drop-in positional module.

**The structure-model convention of 200 does not transfer.** With five chains, a
200-index break offset puts every layout past ESM-2's trained range of 1026, so
the arm has no in-distribution examples at all. Offset 100 already loses the
gap-100 cell for the same reason. Any chain-break offset for a 5-chain complex
on a 1026-position backbone has to be roughly 50 or below, which is a constraint
the plan's default would have violated silently.

## 2026-10-10 - The motivating nuisance does not occur at natural chain-length scales

This is the most consequential Phase 2 result and it qualifies the whole
project. H1's motivating example is that a CDR3-beta three residues longer
shifts every downstream cross-chain offset by three. Tested directly, on
complexes matched for allele and peptide and differing only in CDR3-beta
length, the baseline shows no length-attributable effect.

A0 sensitivity is 0.0039 nats per residue [0.0025, 0.0053]. Against a null with
lengths shuffled within each set, the excess is 0.0003 [-0.0013, 0.0018]. The
interval contains zero. The measured slope is within-set variation driven by TCR
content, mostly CDR3-alpha, which no positional scheme can remove. Every arm's
ratio interval against A0 contains 1 on the primary sample.

**This is consistent with Phase 1, not a contradiction.** Phase 1 measured GSI
against injected gaps of 1 to 150 and found the effect large from gap 25 upward
(GSI 7 to 12) and at or below 1.8 at gaps of 1 to 5. Natural CDR3-beta length
varies by a median of 3 to 4 residues within a matched set, which corresponds to
the small-gap regime where Phase 1 already showed the effect is weak. The
artifact is real and large when the offset is large. Real receptors do not
produce offsets that large.

**Consequence for the project's framing.** The Phase 1 headline stands as
measured: ESM-2 is strongly sensitive to cross-chain position offsets. The
practical motivation for fixing it is much weaker than H1 assumed, because the
naturally occurring perturbation sits in the regime where the model is close to
insensitive. A reader deciding whether to adopt a chain-aware positional module
on this evidence should decline.

The caveat that keeps this from being final: TCR chains here are CDR3 loops
only, so length variation is a few residues. Full variable domains vary far
more, and a model concatenating full chains would see larger offset shifts. That
is a reason to retest with full-length chains, not a reason to discount this
result, and it was already recorded as a limitation before this measurement.

## 2026-10-10 - Permutation sensitivity does fall, at a cost, and A2 is near-circular again

Chain order permutation reorders tokens, so the module cannot erase it by
construction in the way it erased the injected gap.

| Arm | Median abs score change | Ratio vs A0 | Usable |
|---|---|---|---|
| A0 | 0.561 [0.530, 0.590] | 1 | yes |
| A2 reset | 0.0023 [0.0020, 0.0026] | 0.004 | no |
| A3 offset 25 | 0.278 [0.251, 0.300] | 0.49 | yes |
| A3 offset 50 | 0.114 [0.104, 0.130] | 0.20 | no |
| A2+A3 offset 25 | 0.428 [0.403, 0.457] | 0.76 | no |
| A2+A3 offset 50 | 0.111 [0.099, 0.131] | 0.20 | no |

The reduction is real: a permutation moves the baseline score by 0.56 nats and
every arm reduces that.

Two qualifications. A2's near-zero is close to circular for the same reason as
before: with per-chain reset a permuted layout presents the model with the same
(token, position) multiset, so near-invariance follows from the construction
rather than from anything about attention. And only A0 and A3 at offset 25 clear
the 0.25 usability bar. The arms with the largest reductions are the ones that
have degraded into unusable scorers, which is the tension that training is
supposed to resolve.

## 2026-10-11 - Gate G2 failed on A4 and the arm was killed, as pre-specified

The plan's G2 check runs at 10% of steps on the full-module arm and kills the
run if nuisance sensitivity is not already below the baseline's. A4 failed two
of three checks at step 30 and was stopped.

Loss fell from 2.95 to 2.36 with no NaN, so that check passed. The held-out
allele MHC margin went to -0.26, below the uniform prior, so the catastrophic
forgetting check failed. Median absolute slope of MHC score against CDR3-beta
length was 0.0155 against A0's 0.0034 at the same step, so sensitivity was
higher, not lower.

This is the same degradation measured zero-shot: the module presents positions
the backbone has never seen, and 30 LoRA steps do not repair MHC scoring. The
gate was written to catch exactly that and it did. The arm is reported as killed
with no final metrics rather than being revived with a longer budget, because
the rule was fixed before the run.

## 2026-10-11 - Training with the module is indistinguishable from baseline at this scale

ESM-2 35M, CPU, 300 steps at batch 16 over 4,800 examples, LoRA r=16 on
attention and FFN, identical batches, masks, optimizer and schedule across arms,
trainable parameters matched at 1,661,760 by padding the arms without a chain
embedding.

| Arm | Held-out allele MLM loss | Delta vs A0 | Held-out epitope MLM loss | MHC margin |
|---|---|---|---|---|
| A0 | 2.443 | n/a | 2.451 | -0.024 |
| A1 embedding | 2.345 | -0.098 | 2.347 | -0.091 |
| A2 reset | 2.608 | +0.166 | 2.622 | -0.071 |
| A3 offset 50 | 2.329 | -0.113 | 2.361 | -0.119 |
| A4 full | killed at G2 | n/a | n/a | n/a |
| A5 A4 plus augmentation | 2.295 | -0.148 | 2.439 | -0.097 |

Every arm sits near the uniform prior of 2.9957 nats on held-out data while
training loss reaches about 1.0, so the arms are fitting the training set rather
than generalizing. The 0.1 to 0.15 nat advantages of A1, A3 and A5 are small
against that absolute level and rest on one seed, so they are not read as the
module helping.

Two things are firmer. Per-chain reset alone is worse than baseline (+0.166
nats, and the highest length sensitivity of any arm). And every arm including A0
finishes with the MHC margin at or below zero, so LoRA at this scale damages the
scorer regardless of the module.

Length sensitivity does not discriminate between arms, because the shuffled null
tracks the observed slope in every one. A5's lower raw slope comes with a
matching drop in its null floor, so it reflects less score noise rather than less
length dependence. That agrees with the zero-shot finding that there is no
length-attributable effect to remove.

What this does not establish, stated so it is not inferred: H2 remains untested.
These runs are a dev-loop at 35M with one seed, no learning-rate check and no
650M comparison, on a backbone that cannot discriminate binding. The plan's
reported metric for H2 is held-out binding AUROC, which does not exist here. The
arms show the training harness works and the module trains without diverging.
They do not show it helps.

## 2026-10-11 - The plan's 80% coverage target is not enforced, and the deviation is stated

Line coverage on `src/cla` is 58%. The plan asks for 80% enforced in CI. The
gap is concentrated in driver modules: the audit, figures, aggregate,
allele-balanced, cluster-CI and training entry points.

Those drivers are exercised end to end by the runs whose outputs are committed
in `results/`. Covering their remaining lines means unit tests that assert a
script wrote a file, which guard nothing and have to be maintained forever.

The standing testing rule for this work is explicit that coverage is not a goal
and is never a reason to add a test, and that it holds regardless of a repo's
own threshold. That rule governs here, so the target is not met and is not
chased. CI reports coverage and does not gate on it, with the reason written
into the workflow file rather than left for a reader to infer.

This is a deliberate deviation from the plan, recorded as one.

## 2026-10-11 - The 650M Phase 2 training arms are out of reach and are not attempted

The plan's final Phase 2 runs are six LoRA arms at 650M, budgeted at 10 to 15
hours each on an A100-class GPU. On this machine a 650M forward pass over ~320
residues takes 0.46 s, and training adds backward and optimizer cost on a
backend without the kernels an A100 has.

A single arm would run for days and six arms plus extra seeds would run for
weeks, against a plan time-box of 10 days for the whole final phase. The arms
are not started, rather than started and abandoned partway to produce a number
that could not be compared against anything.

Phase 2 training is therefore reported at 35M dev-loop scale only, which is what
the plan itself prescribes for the free-tier stage. The 650M gap is a limitation
of this execution, not a finding about the method.

## 650M allele-balanced run completes; headline holds at both sizes

The 51-allele balanced sample (n = 610, Kish effective sample size 26.86) now covers 650M as
well as 150M. MHC_HEAVY G1 with token dropout on: GSI 4.41 [3.70, 5.30] at gap 25 and
3.27 [2.65, 4.19] at gap 100, cluster bootstrap over alleles. Both lower bounds are far above
1.0, so H1 is supported on an allele-general sample at both admissible sizes, not just the
HLA-A*02:01-dominated one.

Two things change in the report. Limitation 6 no longer says the balanced run covers 150M
only. Section 4.6 gains the balanced comparison, which reproduces the falling-with-scale
direction on 51 alleles rather than 9.

One new caveat: GSI is not monotone in gap at 650M. Gap 100 scores below gap 25, where 150M
keeps rising. One sample at two sizes cannot separate a real non-monotonicity from noise, so
the claim stays at "both sizes exceed 1.0 by a wide margin at gap 25 and above".

Per-allele GSI is now committed (`results/phase1/allele_balanced/per_allele_gsi.tsv`). At
n = 27 per allele, 22 of 51 alleles clear 1.0 on their own lower bound at 150M and 21 of 51
at 650M, with medians 7.07 and 3.90 at gap 25. The panel-level effect is not resolvable
allele by allele, and the figure shows the spread rather than hiding it.

## H2 tested on a binding classifier: not supported

The masked-LM arms could not answer H2, because the plan's metric is held-out binding AUROC and
nothing in those arms predicts binding. A binary classifier head on the same LoRA arms supplies
it: ESM-2 35M, mean-pooled residue representations, 2,400 positives plus one decoy each, 300
steps, three seeds per arm, 1,663,201 trainable parameters in every arm.

Neither A4 (full module) nor A5 (A4 plus chain-order augmentation) beats A0. The seed-mean AUROC
deltas against A0 are -0.061 [-0.103, 0.049] on held-out alleles and +0.104 [-0.143, 0.113] on
held-out epitopes for A4, and +0.044 [-0.080, 0.089] and +0.081 [-0.054, 0.095] for A5. Every
delta interval covers zero, every arm's AUROC interval covers 0.5, and the seed standard
deviation of 0.03 to 0.12 is as large as the deltas.

The result is negative and underpowered at once, and the report says both. A test whose arms all
sit at chance cannot distinguish "the module does not help" from "nothing here predicts binding".
What it does rule out is the specific claim H2 makes, that the module improves held-out binding
AUROC at this scale.

Learning rate was selected on A0 alone over 1e-4, 5e-4, 1e-3 by held-out-allele AUROC. There was
no validation eval, so this is test-set selection. It was applied only to the baseline, which
biases toward A0 and against the hypothesis, so it does not rescue a negative result. Recorded
rather than hidden.

One observation kept as an observation. A0 and A4 fit the training set, final loss 0.16 to 0.32
from 0.69, while held-out AUROC sits at chance. A5, which differs from A4 only by chain-order
augmentation, never fits at all: final loss 0.682, 0.691, 0.485. It also has the highest held-out
mean and the smallest seed spread. That the augmentation removes a layout shortcut the other arms
learn is consistent with these numbers and is not established by them, since the delta still spans
zero. It is not claimed as a finding.

## The 650M infeasibility claim was wrong and is withdrawn

The entry above, "The 650M Phase 2 training arms are out of reach and are not
attempted", is withdrawn. It was an estimate, never a measurement, and the
measurement contradicts it.

A 650M LoRA classifier step on MPS at batch 16 takes 2.87 s, measured over 12
steps. A 300-step arm is therefore about 14 minutes of training plus 5 minutes of
evaluation, so nine runs (three arms, three seeds) take roughly three hours, not
weeks. The error came from extrapolating the 35M masked-LM arms on CPU to 650M on
the GPU backend, two changes at once, without timing either.

The 650M arms are now running. The general lesson is the one this project keeps
finding: an estimate that decides whether to run an experiment should be
replaced by a two-minute measurement before it is written into a document.

## H2 at 650M, and the paired analysis that changed the answer

The 650M classifier sweep ran: three arms, three seeds, 22 minutes per run, 12,176,641
trainable parameters matched across arms. The headline answer to H2 is unchanged. No arm is
an above-chance binding predictor at either size, and the only interval that excludes 0.5 is
A0 on held-out epitopes, which sits below it.

What changed is the reading of the arm differences. The arm means at 35M looked like noise
because the between-seed standard deviation (0.03 to 0.12) swamped them. That comparison was
wrong for this design. Seeds are shared across arms: seed k fixes the training subsample, the
shared initialization and the batch order, so A0 seed k and A4 seed k differ only by the
module. The paired difference is what the experiment produces and the arm-mean difference
throws it away.

Paired by seed, three contrasts agree in sign across all six seeds at both model sizes:

| Contrast | 35M | 650M |
|---|---|---|
| A4, held-out allele | -0.061 | -0.127 |
| A4, held-out epitope | +0.104 | +0.067 |
| A5, held-out epitope | +0.082 | +0.100 |

The module raises held-out-epitope AUROC and the full module lowers held-out-allele AUROC,
across a 19-fold change in backbone size. That split is what the chain-layout argument
predicts: a baseline leaning on layout geometry should generalize worst to antigens it has
never seen.

Four limits are stated with it rather than after it, because any one alone would overstate it.
The baseline is below chance on the shard that improves, so the module moves it toward 0.5
and not above. Three seeds per cell; a sign test on six seeds gives p = 0.031 two-sided and
nothing survives correction for the four contrasts examined. The pairing was chosen after
seeing the unpaired result, and a post-hoc analysis is worth less than a pre-registered one.
Negatives remain mismatched decoys.

H2 as written stays not supported. The directional pattern is recorded as the next experiment
to run, not as a finding: the held-out-epitope shard, validated non-binders, and enough seeds
to resolve a 0.05 AUROC shift.

The analysis lives in `src/cla/eval/h2_paired.py` so it reruns rather than being a number in
prose, with one test covering the failure that would corrupt it silently, a half-finished
sweep pairing mismatched seeds.
