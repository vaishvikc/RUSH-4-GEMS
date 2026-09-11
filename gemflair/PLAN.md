# gemflair — runbook

## 1. What this is

`gemflair` evaluates a pretrained GEM (a 76.9M-parameter decoder-only transformer
trained on tokenized CLIF timelines) as a frozen feature extractor on the FLAIR ICU
prediction benchmark. Raw CLIF tables go in; `clifpy` derives the four tables the
GEM's collation config expects; FLAIR builds four task cohorts, each row carrying a
`feature_cutoff_dttm` prediction point; `cocoa` collates and tokenizes each
hospitalization into one timeline; the **winnower** cuts every timeline at every
prediction point so no token after the cutoff can reach the model; `cotorra` runs the
GEM once over the deduplicated union of cuts and emits one 1024-d state vector per
cut; XGBoost is trained per task on those vectors; `flair build-report` emits a
report bundle per task. The model is never fine-tuned and never sees a label.

```
config/gemflair.yaml            ← the one file a user edits
        │
        ├─ generates ─► work/generated/clif_config.json   (for FLAIR)
        ├─ generates ─► work/generated/collation.yaml     (for cocoa)
        └─ generates ─► work/generated/extraction.yaml    (for cotorra)
        │
  ┌─────▼───────────────────────────────────────────────────────────────┐
  │ 0  prep       clifpy → 4 derived CLIF tables          [once]        │
  │ 1  cohorts    flair load-task ×N → cohorts/<task>.parquet  [once]   │
  │ 2  collate    cocoa collate → meds.parquet             [once]       │
  │ 3  tokenize   cocoa tokenize --tokenizer-home <GEM>    [once]       │
  │                  → tokens_times.parquet                             │
  │ V  verify-rope   reproduce eval_loss, settle position ids  [gate]   │
  │ 4  winnow     ★ NEW ★ cohorts ⋈ tokens_times at cutoffs             │
  │                  → held_out_for_inference.parquet + cut_index.parquet│
  │ 4b audit      leakage + integrity assertions            [gate]      │
  │ 5  extract    cotorra extract → features-held_out-<model>.parquet   │
  │ 6  fit        XGBoost per task → preds/<task>.parquet               │
  │ 7  report     flair build-report ×N → reports/<task>/*.json         │
  └─────────────────────────────────────────────────────────────────────┘
```

Stages 0–3 are task-independent and run once. Stage 4 is the only new algorithm.
Stage 5 is the only GPU stage.

## 2. Run order

Every command runs from the `gemflair/` directory under `uv run`.

```bash
uv run gemflair prep        -c config/gemflair.yaml   # stage 0
uv run gemflair cohorts     -c config/gemflair.yaml   # stage 1
uv run gemflair tokenize    -c config/gemflair.yaml   # stages 2 + 3
uv run gemflair winnow      -c config/gemflair.yaml   # stage 4 + audit
uv run gemflair verify-rope -c config/gemflair.yaml   # ★ GATE ★
uv run gemflair extract     -c config/gemflair.yaml   # stage 5 (GPU)
uv run gemflair fit         -c config/gemflair.yaml   # stage 6
uv run gemflair report      -c config/gemflair.yaml   # stage 7
```

`uv run gemflair run` chains stages 0–7 in that order. It does **not** include
`verify-rope`, because that gate needs a human to read its output.

### ★ The `verify-rope` gate ★

**`verify-rope` must be run after `tokenize` (and after `winnow`, which writes the
`subject_splits.parquet` it reads) and before any full `extract`. Do not skip it.**

The `time_based_rope` setting is **not published with the GEM weights** (see §5.3).
`cotorra` builds position ids two ways, and choosing wrong silently degrades every
representation without raising an error:

```
  time_based_rope present:  position_ids = s_elapsed_past / sec_per_pos_id + arange(n)
  time_based_rope absent:   position_ids = arange(n)
```

`verify-rope` computes mean next-token cross-entropy on the `tuning` split under both
schemes and compares them to the eval loss recorded in the published
`trainer_state.json`, **2.256498098373413**. It writes `work/audit/verify_rope.json`
with `target`, `with_rope`, `without_rope`, and `selected`.

Read the result:

- A loss landing near 2.2565 confirms the scheme **and** confirms that collation and
  tokenization reproduce the original pipeline.
- If neither lands near the target (different MIMIC snapshot, different split), the
  **lower** loss still identifies the scheme.
- If `selected` is `plain`, remove the `model.time_based_rope` block from
  `config/gemflair.yaml` (set it to `null`) and re-run `verify-rope` to confirm the
  generated `extraction.yaml` no longer carries it. Then extract.

Because `verify-rope` needs `subject_splits.parquet`, the practical order on a fresh
site is: `prep → cohorts → tokenize → winnow → verify-rope → extract → fit → report`.

## 3. Laptop smoke test

```bash
cd gemflair
sed -i '' 's/max_encounters_per_task: null/max_encounters_per_task: 200/' config/gemflair.yaml
uv run gemflair run -c config/gemflair.yaml
```

Expect `work/reports/<task>/` for all four tasks, each with two JSON files, and every
file under `work/audit/` reporting `"ok": true` on every check. **Revert the config
edit afterwards.**

`max_encounters_per_task` subsamples *encounters* (not rows) per split with the
XGBoost seed, so the smoke test is deterministic. Stages 0 and 2 dominate the wall
clock on a laptop, not the forward passes.

## 4. HPC runbook

`extract` is the only stage that needs a GPU. Everything else is CPU and I/O; stage 0
is memory-hungry (`clifpy` over ~430 MB of labs) and stage 2 is the other slow one.

Expect a few GPU-hours on one A100 for a full four-task run: the two daily tasks emit
one row per 07:00 window per stay, so the deduplicated union is estimated in the high
hundreds of thousands of rows at up to 4096 tokens each.

```bash
#!/bin/bash
#SBATCH --job-name=gemflair-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
cd $SLURM_SUBMIT_DIR/gemflair
uv run gemflair prep     -c config/gemflair.yaml
uv run gemflair cohorts  -c config/gemflair.yaml
uv run gemflair tokenize -c config/gemflair.yaml
uv run gemflair winnow   -c config/gemflair.yaml
```

```bash
#!/bin/bash
#SBATCH --job-name=gemflair-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
cd $SLURM_SUBMIT_DIR/gemflair
uv run gemflair verify-rope -c config/gemflair.yaml   # read the JSON before continuing
uv run gemflair extract     -c config/gemflair.yaml
uv run gemflair fit         -c config/gemflair.yaml
uv run gemflair report      -c config/gemflair.yaml
```

Run the two GPU-job lines as two submissions with a human read of
`work/audit/verify_rope.json` in between; chaining them blindly defeats the gate.

Every stage is idempotent and re-entrant: `prep` skips derived tables that already
exist (pass `force=True` to rebuild), and every later stage overwrites its own
outputs. Each stage re-runs the cheap startup audit and exits non-zero if it fails,
so a failed gate stops the SLURM script.

Tune `extract.batch_size` for the GPU. `shard_size` is deliberately absent from the
generated `extraction.yaml` and must stay absent: sharding interleaves rows and
breaks the positional join back to `cut_index.parquet`.

## 5. Known limitations

### 5.1 From the design spec (§14), verbatim

1. **MIMIC-on-MIMIC is a plumbing check, not a generalization result.** The GEM was
   pretrained on MIMIC and FLAIR's test split contains patients it saw. Numbers from
   this configuration establish that the pipeline works. Evaluation means pointing
   the same config at a non-MIMIC CLIF site with the same GEM.
2. **Stitched encounters lose cross-hospitalization context.** Per §3.1 the model
   consumes one hospitalization; FLAIR may stitch several into one encounter. The
   audit reports how many rows are affected.
3. **Truncation at `max_len`.** Stays longer than 4096 tokens contribute only their
   most recent 4096 tokens. The audit reports the affected fraction.
4. **Vocabulary transfer.** At a non-MIMIC site, codes absent from the frozen
   vocabulary become `UNK`. The audit reports the rate; a high rate weakens every
   representation.
5. **Rows removed for `n_past == 0`** shrink the report denominator relative to the
   cohort FLAIR built. Counted and reported in `audit/winnow.json`.

### 5.2 SOFA is stamped at end-of-stay

`clifpy`'s SOFA function returns one row per hospitalization with **no timestamp**, so
`prep` sets `event_time` from `discharge_dttm`. Every prediction point is at or before
discharge, so SOFA's 23 vocabulary tokens are excluded from the model's context at
essentially every prediction point.

This is **not a leak** — the token is dropped by the winnower, not admitted — and it
matches how the GEM was trained, which also only ever saw SOFA as a terminal token.
But it means those 23 tokens are effectively unused, and **nothing in the pipeline
flags it**. If a future `clifpy` gains a time-resolved SOFA, the derived table gains
real event times and those tokens start contributing.

### 5.3 `clif_sofa` fails all-or-nothing on a dataset with active admissions

`event_time` comes from `discharge_dttm`, which is **null for a still-admitted
encounter**, and `prep.check` rejects null time values. MIMIC has zero nulls across
546,028 hospitalizations, so this is not currently triggered — but a site with live
patients will hit a hard stop at stage 0.

That is deliberate. Silently dropping those rows is exactly the failure mode this
project exists to prevent. The operator must decide the policy explicitly: filter
still-admitted encounters out of the cohort, or impute a censoring time, or accept a
modified `prep` that excludes them from `clif_sofa` only. There is no default.

### 5.4 The `time_based_rope` setting is not published with the weights

`tokenizer.yaml` records tokenization settings only; `training_args.bin` is a
HuggingFace `TrainingArguments` and does not carry `time_based_rope`. The setting is
settled **empirically** by `gemflair verify-rope` against the recorded eval loss
`2.256498098373413` in `trainer_state.json`. See the gate in §2. **It must be run
before any full extraction.**

### 5.5 `held_out` is a filename slot, not a statistical split

`cotorra` names its output by split and extracts only splits that have a
`<split>_for_inference.parquet`. The entire deduplicated union of cut points — both
FLAIR train and FLAIR test — is therefore written to `held_out_for_inference.parquet`
and extracted once. The real train/test split lives in each cohort's own `split`
column and is applied at the XGBoost stage (`fit.py`).

`subject_splits.parquet` is written alongside it only because `cotorra`'s
`Loader.__init__` unconditionally derives `<split>_tokens_times.parquet`. Extraction
ignores its split values. `verify-rope` does read its `tuning` split, which is why
`winnow` carves one deterministically out of train (10%, seeded with
`xgboost.seed`).

## Measured timings

These are the only real numbers we have so far, taken from a **partial pipeline
run** on full MIMIC data on an Apple M-series CPU. Stages `tokenize`, `winnow`,
`extract`, `fit`, and `report` have **not** been timed on real data yet — treat this
section as covering stages 0 and 1 only, not the whole pipeline.

| Stage | Measured cost | Notes |
|---|---|---|
| `gemflair prep` | **2h 10m total** | medications ~3 min, respiratory support ~72 min, SOFA ~55 min. Peak memory roughly 6.5 GB. |
| `gemflair cohorts` | **~77 min for one task** (`icu_daily_mortality`) | four tasks ⇒ roughly 5 hours total |

### `max_encounters_per_task` does not make `cohorts` cheaper

This is easy to get backwards, so it is stated plainly: setting
`max_encounters_per_task` does **not** reduce the cost of the `cohorts` stage.
`flair load-task` builds the complete cohort first, and only afterward does gemflair
subsample it. The cap shrinks every stage downstream of cohort building — collate,
tokenize, winnow, extract, fit — but not cohort building itself. Budget the full
~77 min/task (~5h for four tasks) regardless of the cap.

### Resumability

Both `prep` and `cohorts` skip work that is already present on disk, so an
interrupted run resumes rather than restarting from scratch. To force a rebuild of a
specific piece, delete the corresponding file under `work/clif_derived/` (for `prep`)
or `work/cohorts/` (for `cohorts`) and rerun.

### Where the GPU comes in

Only the `extract` stage uses a GPU. Every stage before it (`prep`, `cohorts`,
`collate`, `tokenize`, `winnow`) is CPU- and memory-bound and can be run on a laptop
given enough time and RAM.

## 6. Open items

Copied from the design spec (§15), verbatim.

| Item | Resolution path |
|---|---|
| `time_based_rope` setting and `sec_per_pos_id` | `gemflair verify-rope` against the recorded 2.2565 eval loss; confirm with the weights' authors if the run is ambiguous |
| Whether `clifpy` reproduces the exact derived tables used for GEM training | Compare `UNK` rate and vocabulary coverage after stage 3; a low `UNK` rate is evidence the derivation matches |
| True cohort sizes for the two daily tasks | Measured at stage 1 on the first run; determines whether `max_encounters_per_task` is needed on HPC |
