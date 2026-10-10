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
