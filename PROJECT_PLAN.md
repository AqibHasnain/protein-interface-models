# Chain-Layout Nuisance Sensitivity in Concatenation-Based Protein Interface Models

**An executable project plan.**

Audience: an autonomous coding agent with shell access, a GitHub account, and a human collaborator who can provision cloud accounts.

---

## 0. The one-paragraph version

Sequence-level models of multi-component biological complexes (TCR–pMHC, antibody–antigen, peptide–MHC) are built by concatenating the component chains into a single string and running a protein language model over it. The relative-position geometry that the model sees therefore depends on how long each chain happens to be — a CDR3β that is three residues longer shifts every downstream peptide↔MHC relative offset by three, with no change to the underlying biology. Structure models solved this years ago by inserting a large residue-index offset at chain breaks; sequence-level interface models mostly did not. **Phase 1 of this project measures whether that omission leaks, using inference only and effectively zero budget. Phase 2 fixes it with a drop-in change to the positional embedding and measures the gain under matched conditions.** The project is deliberately structured so the expensive half only happens if the cheap half returns a positive signal.

The intellectual template is Spatial Transformer Networks (Jaderberg et al., NeurIPS 2015): identify a nuisance transformation group that an existing architecture handles by accident rather than by design, and fix it with a module that drops into existing infrastructure without changing the loss, the optimizer, or the parameter count.

---

## 1. Scientific claims to be tested

State these up front so the project cannot drift.

**H1 (the audit).** For a concatenation-based interface model, applying a *biologically null relayout* — permuting chain order, or changing the number of separator tokens between chains — changes the model's binding score by an amount that is non-trivial relative to the binder/non-binder score separation.

**H2 (the fix).** Restoring chain-relative positional structure (per-chain position reset + learned chain-identity embedding + large relative-position offset across chain breaks) reduces that nuisance sensitivity and does not hurt, and plausibly improves, held-out generalization on strict allele and epitope splits.

**H3 (the mechanism, secondary).** The gain from H2 is concentrated in examples where chain lengths deviate most from the training-set mode — i.e. it is a distribution-shift fix, not a capacity fix.

Non-claims, to be stated explicitly in the final report: this project does **not** claim improved TCR design, does **not** claim state-of-the-art binding prediction, and does **not** make any wet-lab-validated claim.

---

## 2. Compute strategy

The agent **cannot create accounts, accept terms of service, or enter payment details.** All account provisioning is a human task. The agent's job is to prepare exact instructions, then use credentials the human supplies via environment variables.

### 2.1 Tiered plan

| Tier | Platform | Capacity | Cost | Used for |
|---|---|---|---|---|
| 0 | Local CPU / this container | — | $0 | Repo, tests, data prep, tokenizer logic, all unit tests, smoke runs on a 35M model |
| 1 | Kaggle Notebooks | ~30 GPU-hr/week, P100/T4 16GB, 9–12hr sessions, resets weekly | $0 | **Phase 1 audit in full.** Inference-only scoring of ESM-2 650M |
| 2 | Google Colab free | ~15–30 GPU-hr/week, T4, throttled at peak | $0 | Overflow / interactive debugging |
| 3 | Lightning AI Studio | ~80 GPU-hr/month, persistent VS Code environment, phone verification | $0 | Longer LoRA runs that don't fit a Kaggle session |
| 4 | Modal Starter | $30/month in credits, per-second billing, code-first | $0 | Phase 2 training at 650M; parallel ablation fan-out |
| 5 | Modal paid overflow / RunPod / Vast.ai | pay-as-you-go | **$300 cap** | Final matched runs with multiple seeds |

Figures as of October 2026; the agent should re-verify quotas on first run and record what it actually observed in `docs/COMPUTE_LOG.md`, because these tiers change.

**Note for the human:** Modal advertises an academic grant of up to $10k in compute credits for graduate students, labs, and researchers. If you have any academic affiliation, apply before starting — it changes the project's constraints entirely. Also worth checking: Google Cloud's $300 new-user credit and Azure's $200 new-user credit, both of which cover this project outright.

### 2.2 Hard budget rules for the agent

- Maintain `docs/COMPUTE_LOG.md` with one row per GPU job: timestamp, platform, GPU type, wall-clock minutes, estimated cost, git SHA, purpose, outcome.
- Running total must be printed at the top of that file.
- **Stop and ask the human before any single job projected to exceed $25, and before cumulative spend crosses $150 and again at $250.**
- Never leave an instance running idle. Every Modal function gets an explicit timeout. Every notebook session ends with an explicit shutdown cell.
- Prefer Tier 0–3 (free) for anything that is not a final reported number.

### 2.3 Why Modal is the primary paid target

It is code-first and serverless, which suits an agent better than a notebook: jobs are defined in Python files that live in the repo, are version-controlled, run non-interactively, and bill per second with no idle charge. Define every GPU entrypoint as a Modal function in `infra/modal_app.py` so the exact same code path can run locally on CPU with a `--device cpu --smoke` flag.

---

## 3. Data

All public. No licensing blockers, but record each source's license in `data/SOURCES.md`.

| Dataset | Use | Notes |
|---|---|---|
| **VDJdb** | Paired TCR–pMHC interactions (~3×10⁴ after filtering) | Primary paired source |
| **Messemaker et al. 2025 functionally validated TCR–pMHC database** (bioRxiv) | Gold-standard binders *and* non-binders for YLQPRTFLL and GLCTLVAML | This is the evaluation set that matters — it has real negatives, not shuffled decoys |
| **MHC Motif Atlas** | pMHC ligandome, allele binding motifs | Phase 2 stage-1 pretraining |
| **OTS (Observed TCR Space)** | ~1.5×10⁶ paired αβ TCR repertoires | Phase 2 stage-1 pretraining |
| **IMGT / IPD-IMGT-HLA** | MHC allele reference sequences | Needed to materialize full MHC chain sequences from allele names |
| **ESM-2 checkpoints** (`esm2_t12_35M`, `esm2_t30_150M`, `esm2_t33_650M`) | Base models | Via HuggingFace |

**Decoy policy.** Negatives must come from the functionally validated set wherever possible. Where shuffled or repertoire-sampled negatives are unavoidable, generate them with a fixed seed, document the procedure, and report results with and without them separately. Shuffled negatives inflate AUROC and the final report must not quietly rely on them.

**Splits.** Hold out, completely and from everything:
- Class I alleles: a fixed list of 10, recorded in `configs/splits.yaml`
- Epitopes: `YLQPRTFLL` (SARS-CoV-2 Spike) and `GLCTLVAML` (EBV BMLF1), plus every TCR that recognizes them

Write a test (`tests/test_splits.py`) that asserts zero leakage: no held-out allele string and no held-out epitope string appears anywhere in any training shard. Run it in CI. Leakage is the single most likely way this project produces a wrong result.

---

## 4. Repository layout

```
chain-layout-audit/
├── README.md                    # what, why, how to reproduce in 3 commands
├── pyproject.toml               # uv or poetry; pinned deps
├── Makefile                     # make test | make audit | make train | make report
├── .github/workflows/ci.yml     # lint + typecheck + unit tests + leakage test on every push
├── configs/
│   ├── base.yaml                # shared defaults
│   ├── splits.yaml              # held-out alleles and epitopes (single source of truth)
│   ├── audit/*.yaml             # Phase 1 configs
│   └── train/*.yaml             # Phase 2 configs, one per arm
├── src/cla/
│   ├── data/
│   │   ├── sources.py           # download + checksum verify, idempotent
│   │   ├── assemble.py          # allele name -> full MHC sequence; chain assembly
│   │   ├── splits.py            # deterministic split logic
│   │   └── dataset.py           # torch Dataset, returns chain-annotated records
│   ├── layout/
│   │   ├── spec.py              # ChainSpec: ordered chains + roles + lengths
│   │   ├── concat.py            # BASELINE: naive concatenation
│   │   ├── chain_aware.py       # TREATMENT: per-chain reset + chain embedding + break offset
│   │   └── perturb.py           # the null relayout operators (Phase 1 core)
│   ├── model/
│   │   ├── backbone.py          # ESM-2 loading, LoRA injection
│   │   ├── positional.py        # the actual drop-in module
│   │   └── scoring.py           # span pseudo-log-likelihood (sPLL)
│   ├── eval/
│   │   ├── metrics.py           # NSI, AUROC, AUPRC, bootstrap CIs
│   │   ├── audit.py             # Phase 1 driver
│   │   └── benchmark.py         # Phase 2 driver
│   └── utils/
│       ├── logging.py           # structured JSON + W&B
│       ├── seeding.py           # full determinism helper
│       └── manifest.py          # run manifest: git SHA, config hash, data checksums
├── infra/
│   ├── modal_app.py             # Modal entrypoints
│   └── notebooks/               # thin Kaggle/Colab wrappers that `pip install -e .` and call CLI
├── tests/                       # pytest; see §8
├── results/                     # committed: metrics JSON, figures, tables
└── docs/
    ├── DECISIONS.md             # append-only decision log
    ├── COMPUTE_LOG.md           # every GPU job
    └── REPORT.md                # the final writeup
```

**Rule:** notebooks contain no logic. A Kaggle notebook is three cells — install, set env, call `python -m cla.eval.audit --config ...`. Everything testable lives in `src/`.

---

## 5. Phase 0 — Foundation (target: 3 days, $0)

Deliverables:

1. Repo scaffolded, CI green, `make test` passes on CPU.
2. `cla.data.sources` downloads every dataset, verifies SHA256 against a committed manifest, and is idempotent (re-running is a no-op).
3. `cla.data.assemble` turns a VDJdb row into a `ChainSpec`: an ordered list of `(role, sequence)` where role ∈ {TCRA, TCRB, PEPTIDE, MHC_HEAVY, B2M}. Allele names resolve to full sequences via IMGT.
4. `cla.layout.concat` renders a `ChainSpec` to token IDs the naive way.
5. `cla.model.scoring.span_pll` implements span pseudo-log-likelihood: mask an entire functional span at once, average log-probs over the masked positions.
6. Smoke test: score 50 complexes with `esm2_t12_35M` on CPU in under two minutes.

**Gate G0:** the full pipeline runs end to end on 50 examples on CPU, deterministically (same seed → bitwise-identical scores). Do not proceed until this holds.

---

## 6. Phase 1 — The audit (target: 1 week, $0, Kaggle/Colab)

This is the phase that justifies the project. It is inference-only.

### 6.1 The null relayout operators

Each must be provably biology-preserving. Implement in `cla.layout.perturb`, each with a docstring arguing *why* it is null, and a unit test asserting the multiset of (role, sequence) pairs is unchanged.

| ID | Operator | Why null |
|---|---|---|
| **P1** | Permute chain order in the concatenation | The complex is a set of chains; serialization order is a representational artifact |
| **P2** | Vary separator/linker token count (0, 1, 5, 25) | Separators encode nothing biological |
| **P3** | Pad with *native* framework residues normally truncated | Residues genuinely present in the real protein; truncation is a preprocessing choice |
| **P4** | Insert a fixed-length neutral spacer before the peptide | Shifts downstream relative offsets without changing any chain |
| **P5** | Length-matched control (negative control) | Resample a chain to a different length *with* a real sequence change — expect a real score change. If P1–P4 move scores as much as P5, something is badly wrong |

P5 is essential. Without a positive control you cannot distinguish "model is layout-sensitive" from "scoring code is noisy."

### 6.2 Metrics

Define precisely in `cla.eval.metrics` and test each against hand-computed fixtures.

**Nuisance Sensitivity Index (NSI)** — the headline number:

```
NSI = median_over_examples( |sPLL(x) − sPLL(perturb(x))| ) / |median sPLL(binders) − median sPLL(non-binders)|
```

Interpretation: NSI = 1.0 means a biologically meaningless relayout moves the score as much as the entire binder/non-binder signal. Report NSI per operator, with bootstrap 95% CIs over examples.

Supporting metrics:
- **Rank stability:** Spearman ρ and Kendall τ between original and perturbed score rankings
- **ΔAUROC:** AUROC on the validated binder/non-binder set, original vs. perturbed layout
- **Decision flip rate:** fraction of examples that cross the median threshold under perturbation
- **Length-stratified NSI:** NSI bucketed by CDR3β length, testing H3

### 6.3 Models to audit

Audit at least three, so the finding is about the *paradigm* and not one checkpoint:
- `esm2_t33_650M` with naive concatenation (your own baseline)
- `esm2_t30_150M` with naive concatenation (scale control)
- At least one published concatenation-based interface model with public weights. **Time-box the search for one to four hours.** If no public checkpoint is obtainable in that window, proceed with the ESM-2 baselines only and say so plainly in the report. Do not reimplement anyone's model.

### 6.4 Gate G1 — the go/no-go

This is the most important decision point in the project. Pre-register the threshold now; do not move it after seeing data.

**PROCEED to Phase 2 if, for at least two of P1–P4, on at least two models:**
- NSI ≥ 0.25, **and**
- Spearman ρ between original and perturbed rankings ≤ 0.90, **and**
- P5 (positive control) shows larger movement than P1–P4 (sanity)

**STOP and pivot if:** NSI < 0.15 across all null operators and ΔAUROC < 0.02. The models are already layout-robust; the premise is wrong. This is a real, publishable negative result — write it up as a short robustness note and move to the pivot in §11.

**AMBIGUOUS (0.15 ≤ NSI < 0.25):** escalate to the human with the numbers. Do not self-authorize spending on Phase 2.

### 6.5 Deliverable

`results/phase1/` containing: `nsi.json`, per-operator figures, a length-stratified plot, and a two-page `docs/AUDIT.md`. **This is a standalone artifact.** Even if the project stops here, it is worth showing.

---

## 7. Phase 2 — The fix (target: 3 weeks, ≤$300, Modal + Lightning)

Only runs if G1 passed.

### 7.1 The module

In `cla.model.positional`, implement `ChainAwarePositions` with three independently ablatable components:

1. **Per-chain index reset** — position index restarts at 0 for each chain
2. **Chain-identity embedding** — a learned vector per role (TCRA, TCRB, PEPTIDE, MHC_HEAVY, B2M), added to the token embedding
3. **Chain-break offset** — a constant (default 200, following the structure-model convention) added to the index at each chain boundary; optionally learned per chain-pair type

Keep it under ~150 lines. If it grows past that, the design has gone wrong. It must be switchable by config alone, with no changes anywhere else in the codebase — that drop-in property *is* the contribution.

### 7.2 Training arms

All arms: identical data, identical optimizer, identical schedule, identical LoRA rank, identical step count, matched trainable parameter count (pad the baseline with an equivalently sized unused embedding if needed, and say so).

| Arm | Description |
|---|---|
| A0 | Baseline: naive concatenation + LoRA |
| A1 | + chain-identity embedding only |
| A2 | + per-chain index reset only |
| A3 | + chain-break offset only |
| A4 | **Full module** (all three) |
| A5 | A4 + layout augmentation during training (train on randomly permuted layouts) — tests whether augmentation alone suffices, which would be a cheaper fix and an honest alternative hypothesis |

Run A0 and A4 with 3 seeds each. Run A1–A3, A5 with 1 seed. Report seed variance.

### 7.3 Training recipe

- **Dev loop:** `esm2_t30_150M`, 10% data subsample, on free tiers. Get every arm running end to end before spending a cent.
- **Final:** `esm2_t33_650M`, LoRA (r=16, attention + FFN), bf16, gradient checkpointing, max length 512.
- **Two stages**, mirroring the field's convention: stage 1 on OTS + MHC Motif Atlas with component-specific masking; stage 2 on VDJdb paired data with joint CDR+peptide masking.
- Budget per final run: ~10–15 hr on an A100-class GPU, ~$20–30. Six arms + extra seeds ≈ 9 runs ≈ $200–270. This is why the cap exists.

### 7.4 Gate G2 — mid-training checks

At 10% of stage-2 steps on arm A4, check:
- Loss is decreasing and not NaN
- Zero-shot held-out-allele AUROC has not collapsed (catastrophic forgetting check — the continual-pretraining literature says to watch for this, so watch for it)
- NSI measured on the partial checkpoint is already lower than A0's

If NSI is **not** lower at this point, kill the run. The module either isn't wired in or doesn't work, and 10 more hours will not fix it.

### 7.5 Reported metrics

For every arm:
- NSI per operator (primary — this is what the fix targets)
- Rank stability under perturbation
- Held-out-allele zero-shot pMHC AUROC and AUPRC
- Held-out-epitope TCR recognition AUROC and AUPRC, on the functionally validated set
- Length-stratified AUROC (H3)
- Trainable parameter count, wall-clock training time, inference latency vs. baseline
- Bootstrap 95% CIs on every AUROC; seed std where multiple seeds exist

Present the headline as a single table with A0 and A4 side by side, matched parameters, and the delta with a CI. That is the Spatial-Transformer-paper table format, and it is the right one.

---

## 8. Engineering standards

### 8.1 Testing (`pytest`, CI on every push)

- **Unit:** every perturbation operator preserves the (role, sequence) multiset; sPLL matches a hand-computed fixture on a 5-residue toy case; split logic is deterministic across runs.
- **Leakage:** `test_splits.py` asserts no held-out allele or epitope string appears in any training shard. **This test is non-negotiable and must run in CI.**
- **Invariance:** after the fix, `ChainAwarePositions` produces identical embeddings under P1 chain permutation (this is a property the module should provably have — assert it, don't hope for it).
- **Regression:** golden-file test — 20 fixed complexes, scores pinned to a committed JSON with tolerance 1e-4.
- **Smoke:** full pipeline on 50 examples, CPU, 35M model, under 2 minutes. Runs in CI.
- Target: ≥80% line coverage on `src/cla/`, enforced in CI.

### 8.2 Logging

- **Structured JSON** to `runs/{run_id}/events.jsonl` — every config value, every metric, every checkpoint path.
- **Weights & Biases** free tier for curves and run comparison. Fall back to TensorBoard if W&B credentials aren't supplied; never hard-fail on a missing logger.
- **Run manifest** written at the start of every run: git SHA, dirty-tree flag, full resolved config, config hash, data file checksums, package versions, GPU type, start time. A result without a manifest is not a result.
- **Refuse to launch a paid job from a dirty git tree.** Hard error.

### 8.3 Reproducibility

- Pinned dependencies, lockfile committed.
- `cla.utils.seeding.set_all_seeds()` covering Python, NumPy, torch, CUDA; `torch.use_deterministic_algorithms(True)` where feasible.
- README reproduces Phase 1 in three commands.
- Figures generated by committed scripts, never by hand.

### 8.4 Decision log

`docs/DECISIONS.md`, append-only. Every non-obvious choice gets a dated entry: what was decided, what the alternatives were, why. This is what makes the project legible to a reviewer and is also how the agent avoids relitigating settled questions.

---

## 9. Rabbit holes — stop signals

The agent must treat each of these as a hard boundary. When one is hit: stop, log it in `docs/DECISIONS.md`, and either take the stated alternative or escalate.

| Rabbit hole | Stop signal | What to do instead |
|---|---|---|
| **Reimplementing a published model** | More than 4 hours spent trying to obtain or reproduce someone's checkpoint | Use your own ESM-2 concatenation baseline and say so. The claim is about the paradigm |
| **Structure prediction** | Any impulse to run AlphaFold, ESMFold, or compute ipTM | Out of scope entirely. No access, no budget, and the literature already reports that ipTM does not reliably predict binding |
| **Generative design** | Any impulse to generate CDR3 sequences | Out of scope. Cannot be validated without wet lab. Prediction only |
| **Beating SOTA** | Comparing against NetTCR/pMTnet as the headline claim | The claim is "removes a measurable artifact at zero inference cost," not "best model." Report baselines as context, not as a target |
| **Data-cleaning perfectionism** | More than 2 days on VDJdb curation | Freeze a documented filter, record what was dropped and why, move on. Note it as a limitation |
| **Building a training framework** | Writing a custom trainer loop, custom DDP, or a config system from scratch | Use HF `Trainer` or PyTorch Lightning, and Hydra or plain YAML + Pydantic. This project's novelty is 150 lines in one module |
| **Hyperparameter search** | Any sweep beyond a 3-point LR check on the dev model | Fixed, documented hyperparameters, identical across arms. A sweep would confound the comparison anyway |
| **MHC Class II** | Any work on Class II | Class I only. Mention Class II as future work |
| **Full fine-tuning** | Considering unfreezing the backbone | LoRA only. Full FT blows the budget and confounds parameter matching |
| **Scale chasing** | Wanting ESM-2 3B | 650M is the ceiling. If the effect needs 3B to appear, it isn't the effect being claimed |
| **Infrastructure polish** | Building a dashboard, a web demo, a CLI framework, or a Docker multi-stage build | A `Makefile` and a README. Nobody evaluating this cares about the demo |
| **Metric proliferation** | More than ~8 reported metrics | The ones in §7.5. More metrics means more opportunities to fool yourself |

### 9.1 Time boxes

| Phase | Box | On overrun |
|---|---|---|
| Phase 0 | 3 days | Escalate; likely a data-assembly problem |
| Phase 1 | 7 days | Escalate with partial NSI numbers |
| Phase 2 dev loop | 7 days | Escalate; do not start paid runs |
| Phase 2 final | 10 days | Report whatever arms completed, with CIs |
| Writeup | 4 days | Ship what exists |

### 9.2 Escalation protocol

When escalating: state the gate or box that was hit, the numbers observed, the two or three options with their cost and time, and a recommendation. Do not ask open-ended questions. Do not spend money while waiting.

---

## 10. Deliverables

1. **Public GitHub repo**, CI green, README reproducing Phase 1 in three commands.
2. **`docs/AUDIT.md`** — the Phase 1 nuisance-sensitivity result. Standalone value.
3. **`docs/REPORT.md`** — 6–8 pages: motivation, the STN framing, method, results tables with CIs, ablations, limitations, negative results.
4. **`results/`** — all metrics as JSON, all figures regenerable from committed scripts.
5. **`docs/COMPUTE_LOG.md`** — full accounting, demonstrating the whole thing ran under $300.
6. **A limitations section that is genuinely critical of the work.** Single tissue of claims, one epitope family, no wet-lab validation, LoRA-only, Class I only, one backbone family. Reviewers and hiring managers both read this section first. Writing it well is more differentiating than any number in the results table.

---

## 11. Pre-registered pivot (if G1 says STOP)

If the audit shows the models are already layout-robust, do not improvise. Write up the negative result as `docs/AUDIT.md` (it is a legitimate robustness contribution — "we checked, and here is the evidence that this class of model is insensitive to serialization order") and switch to:

**Learned assay-context tokens for bioactivity prediction.** ChEMBL activity values from different assays, labs, and readouts are not comparable; standard practice is to filter to a single assay type and discard the rest. Instead, give the model a learned per-assay embedding available at training time and marginalized at inference, letting it train on data that is currently thrown away. Same engineering standards, same budget discipline, same gate structure. Evaluate under scaffold and temporal splits, never random splits. The honest framing is that this transfers the batch-covariate idea from single-cell models (scVI and successors) into small-molecule bioactivity, where it is underused — novelty in the transfer, not the idea.

Reuse the entire repo skeleton. That is partly why it is structured this way.

---

## 12. Week-by-week

| Week | Work | Spend |
|---|---|---|
| 1 | Phase 0: repo, CI, data pipeline, CPU smoke test | $0 |
| 2 | Phase 1: perturbation operators, NSI, audit across models. **Gate G1** | $0 |
| 3 | Phase 2 dev loop: module implemented, all 6 arms running on 150M on free tiers | $0 |
| 4 | Final runs A0, A4 (3 seeds each) on 650M. **Gate G2** at 10% | ~$150 |
| 5 | Ablation arms A1–A3, A5 | ~$100 |
| 6 | Analysis, figures, REPORT.md, README, repo cleanup | $0 |

---

## 13. What success looks like

A repo a stranger can clone and reproduce, containing a clearly-stated hypothesis, a cheap pre-registered test of it, an honest gate that was actually respected, a small fix measured under matched conditions with confidence intervals, and a limitations section that names the weaknesses before anyone else does.

If the headline number turns out to be small, that is fine, and the project is still worth showing — provided the measurement is clean. The thing being demonstrated is judgment about what to measure and when to stop, which is scarcer than the ability to fine-tune a protein language model.
