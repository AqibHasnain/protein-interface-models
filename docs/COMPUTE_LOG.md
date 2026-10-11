# Compute log

**Running total: $0.00**

Every GPU job gets a row. A result with no row here is not a result.

## Platform actually used

The project runs entirely on the developer's local machine: Apple M4 Max, 16
cores, 128 GB unified memory, PyTorch MPS backend. No cloud account was
provisioned and no money was spent.

The plan's tiered cloud strategy (Kaggle, Colab, Lightning, Modal) was not
needed for Phase 1. 128 GB of unified memory holds ESM-2 650M in float32 with
room for batched inference, which is the whole of the Phase 1 workload. Tiers 1
to 5 remain the route for Phase 2 training if it is ever run at 650M scale.

| Timestamp | Platform | Device | Wall-clock min | Est. cost | Git SHA | Purpose | Outcome |
|---|---|---|---|---|---|---|---|
| 2026-10-10T18:00Z | local | Apple M4 Max (MPS) | 12 | $0.00 | 9f13500 | Download and smoke all three ESM-2 checkpoints | OK: 35M/150M/650M all forward on MPS; 650M single forward over 300 residues is 0.46 s |
| 2026-10-10T13:40Z | local | Apple M4 Max (MPS) | 76 | $0.00 | 8039dab | Phase 1 audit, esm2_t12_35M, n=1000, 3 targets, dense gap sweep | Complete; nsi.json and summary.tsv written |
| 2026-10-10T16:00Z | local | Apple M4 Max (MPS) | 183 | $0.00 | 9eec818 | Phase 1 audit, esm2_t30_150M, n=1000, 3 targets, dense gap sweep | Complete |
| 2026-10-10T17:50Z | local | Apple M4 Max (MPS) | 136 | $0.00 | 62e67c0 | Phase 1 audit, esm2_t33_650M, n=250, reduced gap set | Complete |
| 2026-10-10T18:45Z | local | Apple M4 Max (MPS) | 25 | $0.00 | 2f26934 | Context gain with scrambled-context control, 3 models, n=300 | Complete |
| 2026-10-10T18:50Z | local | Apple M4 Max (CPU) | 20 | $0.00 | 2f26934 | Cluster bootstrap over MHC alleles, recomputed intervals | Complete |
| 2026-10-10T19:26Z | local | Apple M4 Max (MPS) | 90 | $0.00 | 2f26934 | Allele-balanced confirmation, 150M, 51 alleles, n=610 | Complete |
| 2026-10-11T04:41Z | local | Apple M4 Max (CPU) | 156 | $0.00 | 372d5ce | Phase 2 LoRA arms A0-A5, 35M, 300 steps, 4800 examples | A0,A1,A2,A3,A5 complete; A4 killed by gate G2 at step 30 |
| 2026-10-10T23:45Z | local | Apple M4 Max (MPS) | 215 | $0.00 | 4d232fc | Allele-balanced confirmation, 650M, 51 alleles, n=610 | Complete |
| 2026-10-11T07:30Z | local | Apple M4 Max (CPU) | 190 | $0.00 | 4d232fc | Phase 2 binding classifier, 35M, arms A0/A4/A5 x 3 seeds plus a 3-point LR check | Complete; H2 not supported |
