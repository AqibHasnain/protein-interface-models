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
