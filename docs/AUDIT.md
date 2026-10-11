# Phase 1 audit: chain-layout sensitivity in concatenated ESM-2

## Finding

When ESM-2 scores a TCR-pMHC complex written as one concatenated string, changing only the
position indices at chain breaks, with every token unchanged, moves the score of the MHC heavy
chain several times further than twenty real amino-acid substitutions do. This holds at 150M and
650M parameters, the two conditions whose estimator passes its pre-set sanity check. It does not
hold for the peptide or the CDR3-beta, where no estimator passes the check and nothing can be
claimed.

The audit is inference only on an untrained backbone. It says nothing about a model fine-tuned for
binding.

## What was measured

ESM-2 positions enter only through rotary embeddings, and the model accepts `position_ids` as an
argument. The audit uses that to build a perturbation with no content change. `input_ids` and
`attention_mask` are held bitwise identical and a constant gap is added to the position index at
chain breaks (G1: every break; G2: before the peptide only). Nullness is `torch.equal` on the token
tensors. Gap 0 returns a median absolute shift of exactly 0.0 nats for every model, target and
operator (`fig5_inertness.png`), which is the control that the operator is inert when it should be.

Because lengths and tokens are unchanged, token dropout's mask-fraction rescaling is identical
before and after, so the shift cannot come from that path.

The score is the pseudo-log-likelihood of one chain (the target span) given the rest of the
complex. The reported statistic is the Geometry Sensitivity Index:

```
GSI = median |score(gap g) - score(gap 0)| / median |score(x) - score(x with 20 real substitutions in a context chain)|
```

GSI 1.0 means a content-free relayout moves the score as far as twenty real mutations. The
denominator is small (0.036 to 0.045 nats for MHC_HEAVY), so raw nats are reported beside each ratio.

## Headline

MHC_HEAVY, G1, token dropout on. MHC intervals are 95% cluster bootstrap over alleles.

| Model | n | Gap | Median shift (nats) | 20-substitution shift (nats) | GSI [95% CI] |
|---|---|---|---|---|---|
| 150M | 1000 | 25 | 0.383 | 0.0446 | 8.59 [6.77, 9.40] |
| 150M | 1000 | 100 | 0.469 | 0.0446 | 10.52 [9.78, 16.11] |
| 650M | 250 | 25 | 0.172 | 0.0356 | 4.83 [1.96, 5.66] |
| 650M | 250 | 100 | 0.181 | 0.0356 | 5.08 [2.39, 5.97] |
| 150M, allele-balanced | 610 | 25 | 0.295 | 0.0424 | 6.96 [6.18, 7.85] |
| 150M, allele-balanced | 610 | 100 | 0.497 | 0.0424 | 11.71 [10.39, 13.19] |

Figures: `fig1_dose_response.png` (GSI against gap), `fig3_absolute_scale.png` (raw nats).

The main sample is dominated by one allele: 14 distinct alleles at 150M, HLA-A*02:01 at 69% of
examples, Kish effective sample size 1.92. The allele-balanced run (51 alleles, at most 27 examples
each, effective sample size 26.86) gives the same result, so the effect does not belong to one allele.
Gap 1 stays below 1.0 in both samples (0.71 and 0.50 at 150M); claims at gaps 1 and 2 are not made.

The disturbed computation is real. Scoring the MHC span in the full complex against the same
complex with each other chain's residues shuffled gives +0.60 nats at 150M and +0.19 at 650M, with
the same sign and similar size with token dropout on and off (`context_gain.json`). The model uses
cross-chain content at these sizes, and the position artifact acts on that computation.

Across the two admissible sizes sensitivity falls from 10.52 to 5.08 at gap 100 (intervals
[9.78, 16.11] and [2.39, 5.97] do not overlap; `mhc_cluster_ci.tsv`) and stays about 5x
at 650M. Two points give a direction, not a scaling law. `fig2_scale.png`.

## Admissibility rule

A target counts as evidence only if the median baseline span score exceeds the uniform-prior score
(-ln 20 = -2.996) by at least 0.25 nats. The threshold was fixed in code before any result was read.
A score near the uniform prior carries almost no information, so movement in it cannot support a claim.

| Target | 35M | 150M | 650M |
|---|---|---|---|
| PEPTIDE | 0.037, fails | 0.039, fails | 0.050, fails |
| TCRB | 0.096, fails | 0.104, fails | 0.089, fails |
| MHC_HEAVY | 0.068, fails | 0.627, passes | 1.016, passes |

Margins are from each run's `nsi.json`. PEPTIDE and TCRB GSI exceed 1.0 in many conditions
(`table1.md`), but those rows are not evidence.

## Negative results

- No binding discrimination. Binder-versus-decoy AUROC at 35M is 0.489 [0.464, 0.515] for the
  peptide and 0.533 [0.507, 0.557] for CDR3-beta. Across all sizes the peptide and CDR3-beta point
  estimates stay at or below 0.533. The highest value anywhere is MHC_HEAVY at 150M: 0.569 with token
  dropout on and 0.616 with it off (`fig4_discrimination.png`), which is a span of a conserved
  protein scored against mismatched decoys and is not interpreted as binding signal.
- ESM-2 35M predicts CDR3-beta no better than a frequency table. Its margin over the uniform prior is
  about 0.10, and a unigram model fitted to the dataset's own CDR3 spans did better in a 200-example
  probe recorded in `docs/DECISIONS.md`.
- No estimator tried (whole-span, windowed at several widths, full pseudo-likelihood) lifts the
  peptide or CDR3-beta margin to 0.25 (same probe).

## What this does not show

- Nothing about a trained interface model. The audit measures the untrained backbone, and the claim
  that would matter is about a fine-tuned one.
- Nothing about peptide or CDR3 scoring, the spans that determine binding.
- No effect size for the 650M model with a large sample: n is 250, with 9 distinct alleles, and a
  balanced 650M run was not complete at time of writing.
- At 650M and gap 100, GSI is 0.57 [0.48, 0.73] with token dropout off, against 5.08 with it on. The
  large 650M gap-100 value is specific to the model as published. At gap 25 the arms agree (4.55 off,
  4.83 on). At 150M the off arm is larger (18.14 at gap 100).
- G2 (gap before the peptide only) stays below 1.0 for MHC_HEAVY at both admissible sizes.
- No wet-lab validation, and decoys are mismatched pairs, not validated non-binders.

## Gate G1

PROCEED. The audit driver returned PROCEED partly on the inadmissible peptide and CDR3-beta targets.
The verdict stands on the admissible MHC_HEAVY conditions alone: GSI 4.83 to 10.52 at gaps 25 and
100, all cluster-bootstrap lower bounds above 1.0 and so above the 0.5 requirement. The
pre-registered NSI could not be computed because binder and decoy scores are not separated, so the
decision rests on GSI, a metric defined after the pilot data was seen.

Full history of claims, corrections and withdrawals: `docs/DECISIONS.md`. Full write-up:
`docs/REPORT.md`.
