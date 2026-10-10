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
