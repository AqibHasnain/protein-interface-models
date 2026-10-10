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
