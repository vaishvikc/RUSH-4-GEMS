# gemflair — design spec

**Date:** 2026-09-10
**Status:** approved for implementation planning
**Repo:** `RUSH-4-GEMS`

---

## 1. Purpose

Evaluate a pretrained generative event model (GEM) as a **feature extractor** on the
FLAIR ICU benchmark.

For every FLAIR prediction point, tokenize the patient's hospitalization up to that
point's `feature_cutoff_dttm`, run the frozen GEM over those tokens, and take the
final hidden state as a 1024-dimensional summary of the patient's state at that
instant. Train XGBoost on the train split's representations and score the test
split. Report through FLAIR's standard report builder.

This is the same shape as `flair_baseline`, with one substitution:

```
  flair_baseline:  CLIF tables ──► hand-engineered features ──► XGBoost ──► FLAIR report
  gemflair:        CLIF tables ──► GEM latent state          ──► XGBoost ──► FLAIR report
```

Everything is config-driven. MIMIC and the `GEM-mimic-icu` weights are the smoke-test
pairing; the same config points at a different CLIF site and a different GEM without
code changes.

### Non-goals

- Training or fine-tuning a GEM. The weights are a third-party artifact used as-is.
- Federated orchestration. `coreopsis` is not used.
- Generative scoring (`cotorra generative-score`). Representation extraction only.
- Beating `flair_baseline`. This spec delivers the pipeline and an honest report.

---

## 2. Components

Five packages sit in the repo. Their roles:

| Package | Role here |
|---|---|
| `FLAIR` | Defines tasks, builds cohorts with labels + `feature_cutoff_dttm` + train/test split, builds reports. **Used.** |
| `cocoa` | CLIF parquet → MEDS-long events → tokenized timelines. **Used** (collate + tokenize; winnow replaced). |
| `cotorra` | Trains GEMs and extracts hidden states. **Used** (extract only). |
| `coreopsis` | Federated training via Flower. **Not used** — single-site inference on existing weights. |
| `flair_baseline` | Reference implementation of the CLIF→XGBoost→FLAIR loop. **Read for patterns, not imported.** |

### The model

`GEM-weights-FL-manuscript/GEM-mimic-icu/mdl-cotorra/`

| Property | Value | Source |
|---|---|---|
| Architecture | `LlamaForCausalLM`, 9 layers, hidden 1024, 8 heads | `config.json` |
| Parameters | ~76.9M (cotorra preset `llama_32`) | `reference/cotorra/config/training.yaml` |
| Vocab size | 1344 | `config.json` |
| BOS / EOS | 90 / 125 | `config.json` |
| `pad_token_id` | absent → set to EOS at load | `extractor.py:52-53` |
| Recorded eval loss | **2.256498098373413** at `checkpoint-11336` | `GEM_-mimic-icu/.../trainer_state.json` |

`GEM_-mimic-icu/` (underscore) is the same checkpoint plus optimizer/scheduler/RNG
state. `GEM-mimic-icu/` is weights + configs only and is what the pipeline loads.
`trainer_state.json` is read from the underscore directory for the eval-loss target.

### The tokenizer

`GEM-weights-FL-manuscript/tokenizer.yaml` — 1344-entry lookup, quantile bins,
and the tokenization config:

```yaml
n_bins: 10
fused: true                    # code and quantile fused: LAB-RES//albumin_Q3
include_numeric_values: false
insert_spacers: false          # no TIME// tokens
insert_clocks: false           # no CLCK// tokens
cocoa_version: 26.5.1
```

Vocabulary by prefix:

```
  LAB-RES 519 │ MED-CTS 415 │ MED-INT 178 │ VTL 97 │ ASMT 88 │ LAB-ORD 45
  RESP    37  │ SOFA    23  │ CRRT    22  │ DSCG 12 │ LABEL 11 │ AGE   11
  XFR-OUT  7  │ XFR-IN   7  │ RACE     7  │ CODE  4 │ ETHN   3 │ ADMN   3
  SEX      2  │ POSN     1  │ + UNK, BOS, EOS
```

Loaded with `Tokenizer.from_yaml(..., done_training=True)`, which sets
`is_training=False` and freezes both the lookup and the bin cutpoints
(`reference/cocoa/src/cocoa/tokenizer.py:337-364`). Any code or value not in the frozen vocabulary
maps to `UNK` (token 0).

---

## 3. Findings from source inspection

These drove the design. Each is load-bearing; each cites its evidence.

### 3.1 Timelines are per-hospitalization, not per-patient

`cocoa`'s collation config sets `subject_id: hospitalization_id` and
`group_id: patient_id`. Any table joined via `reference_key` is clipped to the
reference window (`collator.py:206-212`):

```python
if reference_key is not None:
    df = df.join(self.reference_frame, on=reference_key, how="inner").filter(
        pl.col("time").is_between(
            pl.col(self.cfg["reference"]["start_time"]),   # admission_dttm
            pl.col(self.cfg["reference"]["end_time"]),     # discharge_dttm
```

`group_id` appears only at `collator.py:257` and `:284`, inside split assignment —
it keeps one patient's hospitalizations on the same side of the train/held-out
boundary. It does not widen event scope.

**Consequence:** the winnower keys on `hospitalization_id`. A patient's earlier
hospitalizations are not in the context, because the model was never trained to
see them.

### 3.2 `cotorra`'s extractor truncates from the left

`extractor.py:58` slices `x[:ml]` — the **first** `max_len` tokens. For a timeline
longer than 4096 tokens, the extracted state would be the state at token 4096,
not at the cutoff.

**Consequence:** `gemflair`'s winnower emits `tokens_past` already trimmed to the
**last** `max_len` tokens, making cotorra's slice a no-op. `cocoa`'s stock winnower
never hit this because it thresholds at 24h, where sequences stay short.

### 3.3 Row identity does not survive extraction

`loader.py:88-94` selects only `input_ids` (and `s_elapsed_past` when
`time_based_rope` is configured). Every other column is dropped, so `prediction_id`
cannot ride along inside the inference table.

`extractor.py:112` calls `dset.shard(num_shards=n, index=i)`. HuggingFace's default
is `contiguous=False`, so shard *i* takes rows *i, i+n, i+2n, …* — concatenating
shards in filename order does **not** restore input order.

**Consequence:** `shard_size` is omitted from the extraction config so each split
writes exactly one file in input order, and `gemflair` writes a parallel
`cut_index.parquet` in the same row order carrying the identity columns. Features
join back positionally.

### 3.4 No attention mask is passed, and that is correct

`collate_fn` pads with `pad_token_id`, which equals `eos_token_id` (125), and
passes no `attention_mask`. Padding sits *after* the real tokens; causal attention
cannot reach backwards. `first_eos` (`extractor.py:80-87`) therefore resolves to
the index of the last real token, and the extracted vector is genuinely the state
at the cutoff.

### 3.5 `s_elapsed` is not rebased across packing boundaries

`util.py:32-41` (`batched_iter`) concatenates subjects' token streams into
contiguous `max_seq_len` blocks without rebasing `s_elapsed`. A subject longer than
4096 tokens contributes a second block whose first element carries a large elapsed
value.

**Consequence:** after left-trimming, `gemflair` keeps `s_elapsed_past` measured
from the subject's own first token and does **not** rebase to zero. This matches
training conditions.

### 3.6 The collation config requires four derived CLIF tables

`reference/cocoa/config/collation.yaml` reads `clif_medication_admin_continuous_converted`,
`clif_medication_admin_intermittent_converted`, `clif_respiratory_support_processed`,
and `clif_sofa`. None exist in `clif_m/`. They supply
`MED-CTS`(415) + `MED-INT`(178) + `RESP`(37) + `SOFA`(23) = **653 of 1344 tokens**.

**Consequence:** stage 0 generates them with `clifpy` before collation.

### 3.7 The published weights carry their own eval target

`trainer_state.json` records `eval_loss = 2.256498098373413` at `checkpoint-11336`,
which is `best_model_checkpoint` — the published weights. `training_args.bin`
(HuggingFace `TrainingArguments`) does not record `time_based_rope`, and
`tokenizer.yaml` does not either, because it is a training-side setting.

**Consequence:** stage V (`verify-rope`) reproduces the eval and compares against
2.2565 to determine the position-id scheme empirically. See §8.

---

## 4. Architecture

```
config/gemflair.yaml            ← the one file a user edits
        │
        ├─ generates ─► config/clif_config.json      (for FLAIR)
        ├─ generates ─► config/collation.yaml        (for cocoa)
        └─ generates ─► config/extraction.yaml       (for cotorra)
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

### Single global extraction

The union of `(hospitalization_id, feature_cutoff_dttm)` across all tasks and both
splits is deduplicated and extracted **once**. The two daily tasks
(`icu_daily_mortality`, `icu_daily_ltach`) both stamp 07:00 cutoffs and overlap
heavily, so deduplication is a large saving.

`cotorra` names its output by split and extracts only splits that have a
`<split>_for_inference.parquet`. The entire deduplicated union is therefore written
to `held_out_for_inference.parquet`.

> **`held_out` here is a filename slot required by cotorra, not a statistical split.**
> The real train/test split lives in the FLAIR cohort's own `split` column and is
> applied at stage 6. This must be stated in the config comments, in a module
> docstring in `winnow.py`, and in `PLAN.md`.

`subject_splits.parquet` is still required by `cotorra`'s `Loader.__init__`, which
unconditionally derives `<split>_tokens_times.parquet`. It is written from the union
cohort (FLAIR `train` → `train`, FLAIR `test` → `held_out`, 10% of train encounters
→ `tuning`) so the files are valid. Those three files are not read during extraction.

---

## 5. Data contracts

Artifacts, in dependency order. Paths are relative to `work_dir`.

| Artifact | Producer | Key columns |
|---|---|---|
| `clif_derived/*.parquet` | stage 0 | CLIF schema for the 4 derived tables |
| `cohorts/<task>.parquet` | stage 1 | `hospitalization_id`, `hospitalization_join_id`, `patient_id`, `prediction_dttm`, `feature_cutoff_dttm`, `<label_column>`, `split`, `window_number`, `prediction_id`, demographics |
| `processed/meds.parquet` | stage 2 | `subject_id`, `time`, `code`, `numeric_value`, `text_value` |
| `processed/subject_splits.parquet` | stage 4 | `subject_id`, `split` ∈ {train, tuning, held_out} |
| `processed/tokens_times.parquet` | stage 3 | `subject_id`, `tokens: list[u32]`, `times: list[datetime]` |
| `processed/held_out_for_inference.parquet` | stage 4 | `tokens_past: list[u32]`, `s_elapsed_past: list[f64]` |
| `processed/cut_index.parquet` | stage 4 | `row_ix`, `subject_id`, `feature_cutoff_dttm`, `n_past`, `n_kept` — **same row order** |
| `processed/features-held_out-<model>.parquet` | stage 5 | `features: list[f16]` (1024), same row order |
| `preds/<task>.parquet` | stage 6 | the cohort columns + `y_prob` |
| `reports/<task>/*.json` | stage 7 | FLAIR report bundle |
| `audit/*.json` | stages 4b, V | counts, failures, verification results |

`cut_index.parquet` is the join key between FLAIR identity and GEM features.
Its `row_ix` is positional and must equal the row index in both the inference
table and the features table.

---

## 6. Stage specifications

### Stage 0 — `prep`

Generate the four derived CLIF tables with `clifpy` into `work_dir/clif_derived/`:

- `clif_medication_admin_continuous_converted` — dose unit conversion, with
  `_convert_status` and `med_dose_converted` columns as the collation config expects.
- `clif_medication_admin_intermittent_converted` — same.
- `clif_respiratory_support_processed` — respiratory support waterfall.
- `clif_sofa` — SOFA component and total scores.

Idempotent: skip any table already present unless `--force`. Collation reads from
both `clif_dir` and `derived_dir`.

If `clifpy` cannot produce one of these for the configured dataset, the stage fails
with the table name and the affected token prefixes, so the vocabulary loss is
explicit rather than silent.

### Stage 1 — `cohorts`

For each task in `config.tasks`, shell out to:

```
flair load-task <task> --clif-config config/clif_config.json --out cohorts/<task>.parquet
```

plus `--train-end` / `--test-start` when configured. With `site: mimic` and both
null, FLAIR uses its deterministic MIMIC split
(`cohort/stitch.py:assign_split`, random by `hospitalization_join_id`).

When `limits.max_encounters_per_task` is set, each cohort is subsampled immediately
after `load-task` and before anything downstream reads it. Sampling is by
**`hospitalization_join_id` within each split**, not by row: taking random rows
would split an encounter's windows across the subsample and break the stay-level
grouping that FLAIR's report and stage 6's early-stopping split both rely on.
Encounters are drawn with `numpy.random.default_rng(config.xgboost.seed)` so the
subsample is reproducible. The cap is a smoke-test device; `null` is the real run.

Note that FLAIR's split grain is the **stitched encounter**, so every
`hospitalization_id` inside one encounter lands in the same split. Split
assignment is per task, so the same hospitalization can be `train` for one task and
`test` for another. Stage 6 reads `split` from each task's own cohort; the global
`subject_splits.parquet` is only cotorra plumbing.

### Stage 2 — `collate`

`cocoa collate` accepts exactly one `--raw-data-home`, so stage 2 first builds
`work_dir/raw_view/` — a directory of symlinks to every `*.parquet` in `clif_dir`
plus every table in `derived_dir`. Symlinks, not copies: `clif_labs.parquet` alone
is 430 MB. A name present in both directories resolves to `derived_dir` and the
shadowing is logged.

```
cocoa collate --collation-config config/collation.yaml \
              --raw-data-home work_dir/raw_view \
              --processed-data-home work_dir/processed
```

`config/collation.yaml` is `cocoa`'s shipped default with `default_timezone` set
from `config.data.timezone`. The timezone must equal FLAIR's `clif_config.timezone`;
stage 4b asserts it.

### Stage 3 — `tokenize`

```
cocoa tokenize --tokenizer-home <GEM>/tokenizer.yaml \
               --processed-data-home work_dir/processed
```

The frozen tokenizer applies the GEM's vocabulary and bin cutpoints. Stage 4b
reports the `UNK` rate; a high rate means this dataset's codes diverge from the
GEM's training data and the representations should be read with that in mind.

### Stage V — `verify-rope`

See §8. Runs before any large extraction.

### Stage 4 — `winnow` (the new algorithm)

See §7.

### Stage 4b — `audit`

See §9.

### Stage 5 — `extract`

```
cotorra extract --extraction-config config/extraction.yaml \
                --processed-data-home work_dir/processed \
                --model-home <GEM>/mdl-cotorra \
                --output-home work_dir/processed
```

`config/extraction.yaml` sets `max_seq_len` and `extract.max_len` from
`config.model.max_len`, `extract.batch_size` from config, **omits `shard_size`**
(per §3.3), and includes `time_based_rope` only when configured.

`cotorra` selects `cuda` → `mps` → `cpu` automatically (`extractor.py:44-50`).

### Stage 6 — `fit`

Per task, independently:

1. Join `cohorts/<task>.parquet` to `cut_index.parquet` on
   `(hospitalization_id, feature_cutoff_dttm)` to get `row_ix`. `cut_index`'s
   `subject_id` **is** the cohort's `hospitalization_id` — cocoa's collation config
   sets `subject_id: hospitalization_id` (§3.1), so the two are the same key under
   two names. The join renames rather than mapping.
2. Gather `features[row_ix]`, cast float16 → float32. That is **X**. The
   representation is the only input; no demographic or static columns are added.
3. **y** = the task's `label_column` from its `META`.
4. Rows with `n_past == 0` are excluded from both training and prediction (§7.3).
5. Train on `split == "train"`. Hold out 10% of *training encounters*
   (grain `hospitalization_join_id`, matching FLAIR's own grain) as an
   early-stopping set.
6. Predict on `split == "test"`.
7. Write `preds/<task>.parquet` = the cohort rows **unchanged** plus `y_prob`,
   so `prediction_id` cannot drift.

Parameters: fixed sensible defaults plus `early_stopping_rounds`. No
hyperparameter search — add one only if the fixed defaults prove inadequate.

### Stage 7 — `report`

```
flair build-report preds/<task>.parquet --task <task> \
      --cohort cohorts/<task>.parquet --out reports/<task>/
```

Report mode comes from the task's `META`, so it is not passed.

---

## 7. The cohort winnower

The only genuinely new logic. Modeled on `reference/cocoa/src/cocoa/winnower.py`, but the cut point
comes from data rather than config.

### 7.1 Input

- `tokens_times.parquet` — `subject_id`, `tokens: list[u32]`, `times: list[datetime]`
- `cuts` — the deduplicated union of `(subject_id, feature_cutoff_dttm)` over all
  tasks and both splits, where `subject_id` is the cohort row's `hospitalization_id`

### 7.2 Algorithm

```python
def count_past(tokens_times, cuts):
    # join_asof finds the last event at or before the cutoff; its index + 1 is the count.
    events = (tokens_times.select("subject_id", "times").explode("times")
              .with_columns(ix=pl.int_range(pl.len()).over("subject_id")))
    return (cuts.sort("feature_cutoff_dttm")
            .join_asof(events.sort("times"), left_on="feature_cutoff_dttm",
                       right_on="times", by="subject_id", strategy="backward")
            .with_columns(n_past=(pl.col("ix") + 1).fill_null(0))
            .select("subject_id", "feature_cutoff_dttm", "n_past"))


def cut(tokens_times, cuts, max_len):
    n = count_past(tokens_times, cuts)
    return (tokens_times.join(n, on="subject_id")
            .filter(pl.col("n_past") > 0)
            .with_columns(
                tokens_past=pl.col("tokens").list.head("n_past").list.tail(max_len),
                s_elapsed_past=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                ).list.head("n_past").list.tail(max_len))
            .with_columns(n_kept=pl.col("tokens_past").list.len())
            .sort("subject_id", "feature_cutoff_dttm")
            .with_row_index("row_ix")
            .select(INDEX_COLS + INFERENCE_COLS))
```

`list.eval` cannot reference outer columns (`ComputeError: named columns are not allowed in
eval functions`), so the cutoff comparison uses `join_asof` on an exploded event index
instead. The list slicing that follows is unchanged.

Three deliberate properties:

1. **`<=`, not `<`.** Matches FLAIR's inclusive contract
   (`features/_scope.py`, `time <= end_time`).
2. **`.list.head(n_past).list.tail(max_len)`** — cut at the cutoff *first*, then keep
   the **last** `max_len`. This is the fix for §3.2. Order matters; reversing it
   would reintroduce the bug.
3. **`s_elapsed_past` is measured from the subject's own first token and is not
   rebased after trimming.** This matches §3.5.

### 7.3 No minimum-token floor

There is no `min_tokens` threshold. A deployed model fires for every patient at
every prediction point, so the pipeline does too. A row whose cutoff falls shortly
after admission gets whatever tokens exist — typically the static admission tokens —
and a real prediction.

**`n_past == 0`** is the single exception. That means no event at or before the
cutoff, i.e. the cutoff precedes the patient's own admission tokens. Those rows are
removed from both training and prediction. The count and the affected
`prediction_id`s are written to `audit/winnow.json`, because removing test rows
shrinks the denominator FLAIR's report is computed over and that must be visible.

### 7.4 Output

- `held_out_for_inference.parquet` — `tokens_past`, `s_elapsed_past`
- `cut_index.parquet` — `row_ix`, `subject_id`, `feature_cutoff_dttm`, `n_past`,
  `n_kept`, in **identical row order**

Both are written from a single materialized frame in one pass so the orders cannot
diverge.

---

## 8. RoPE verification

`tokenizer.yaml` records tokenization settings only; `training_args.bin` is
HuggingFace `TrainingArguments` and does not carry `time_based_rope`. The setting is
therefore not published with the weights.

`cotorra` builds position ids two ways (`extractor.py:62-75`):

```
  time_based_rope present:  position_ids = s_elapsed_past / sec_per_pos_id + arange(n)
  time_based_rope absent:   position_ids = arange(n)
```

The GEM's vocabulary has no `TIME//` or `CLCK//` tokens, so under the second scheme
the model receives no temporal information at all. A wrong choice degrades every
representation without raising an error.

`trainer_state.json` records `eval_loss = 2.256498098373413` at the published
checkpoint. `gemflair verify-rope` reproduces that evaluation:

1. Take the `tuning` split from `subject_splits.parquet`.
2. Pack token streams into `max_seq_len` blocks with the same logic as
   `util.batched_iter`.
3. Compute mean next-token cross-entropy under each position-id scheme.

Decision rule:

- **Absolute** — a run landing near 2.2565 confirms the scheme *and* confirms that
  collation and tokenization reproduce the original pipeline.
- **Relative** — if neither matches (different MIMIC snapshot, different split),
  the lower loss still identifies the scheme. Wrong position ids degrade next-token
  prediction substantially.

Writes `audit/verify_rope.json` with both losses, the target, and the selected
setting. Config default is `sec_per_pos_id: 300`, cotorra's own default.

`PLAN.md` lists this as the first thing to run on HPC, before any full extraction.

---

## 9. Audit checks

`gemflair audit` writes `audit/<stage>.json` and exits non-zero on failure.

**Startup assertions** (cheap, run before any expensive stage):

1. `config.json`'s `vocab_size` equals `len(tokenizer.yaml.lookup)`.
2. `generation_config.json`'s BOS/EOS match `tokenizer.yaml`'s `BOS`/`EOS` entries.
3. FLAIR `clif_config.timezone` equals collation `default_timezone`.
4. `model.max_len` ≤ `config.json`'s `max_position_embeddings`.

**Post-winnow assertions:**

5. For every emitted row, `max(times[:n_past]) <= feature_cutoff_dttm`, recomputed
   independently of the winnow query. This is the leakage check the project rests on.
6. `len(tokens_past) == len(s_elapsed_past)` and both `<= max_len`.
7. Every task cohort row resolves to exactly one `cut_index` row, or is reported as
   removed with its `prediction_id`.
8. `held_out_for_inference.parquet` and `cut_index.parquet` have equal row counts.

**Post-extract assertions:**

9. `features-held_out-*.parquet` row count equals `cut_index` row count.
10. Exactly one features file exists — more than one means `shard_size` leaked back
    into the extraction config and row order is no longer trustworthy (§3.3).

**Reported, not asserted** (context for interpreting results):

- `UNK` token rate over the tokenized corpus.
- Count of cohort rows whose `hospitalization_join_id` spans more than one
  `hospitalization_id` — those rows lose the earlier hospitalization's context (§3.1).
- Distribution of `n_past`, and the fraction of rows where `n_past > max_len`
  (i.e. context was truncated).

---

## 10. Configuration

`config/gemflair.yaml` is the only file a user edits.

```yaml
data:
  clif_dir: /Users/sudo_sage/Downloads/work/clif_m
  derived_dir: ./work/clif_derived      # stage 0 output; collation reads both
  timezone: US/Eastern                  # must match collation default_timezone
  site: mimic                           # "mimic" enables FLAIR's deterministic split

model:
  dir: ./GEM-weights-FL-manuscript/GEM-mimic-icu/mdl-cotorra
  tokenizer: ./GEM-weights-FL-manuscript/tokenizer.yaml
  trainer_state: ./GEM-weights-FL-manuscript/GEM_-mimic-icu/mdl-cotorra/trainer_state.json
  max_len: 4096
  time_based_rope:
    sec_per_pos_id: 300                 # null disables; settle with `verify-rope`

tasks:
  - icu_daily_mortality
  - icu_daily_ltach
  - extubation_failure_24h
  - icu_readmission

split:
  train_end: null                       # null + site:mimic → FLAIR deterministic split
  test_start: null

extract:
  batch_size: 64
  device: auto                          # auto | cuda | mps | cpu

xgboost:
  early_stopping_rounds: 50
  seed: 42

limits:
  max_encounters_per_task: null         # e.g. 200 for a laptop smoke test; null = full run

work_dir: ./work
```

`clif_config.json`, `collation.yaml`, and `extraction.yaml` are generated from this
on every run, so they cannot drift out of sync.

---

## 11. Module layout

```
gemflair/
  config/
    gemflair.yaml                 # the user-edited config
  src/gemflair/
    config.py       # load + validate gemflair.yaml; generate the three derived configs
    prep.py         # stage 0 — clifpy derived tables
    cohorts.py      # stage 1 — flair load-task
    tokenize.py     # stages 2+3 — cocoa collate + tokenize (frozen tokenizer)
    winnow.py       # stage 4 — cohort-driven cut  ★ core logic ★
    verify_rope.py  # stage V — eval-loss reproduction
    audit.py        # stage 4b — assertions
    extract.py      # stage 5 — cotorra extract + positional join back
    fit.py          # stage 6 — XGBoost per task
    report.py       # stage 7 — flair build-report
    cli.py          # `gemflair run` and `gemflair <stage>`
  tests/
    test_winnow.py       # cut logic against hand-built fixtures
    test_config.py       # generated-config correctness
    test_audit.py        # each assertion fires on a crafted violation
  PLAN.md
```

Each module owns one stage, reads named artifacts, and writes named artifacts.
No module imports another's internals; `config.py` is the only shared dependency.

---

## 12. Testing

**Unit tests, hand-built fixtures** — the winnower is where a bug is invisible and
fatal, so it gets the most coverage:

- A timeline with known timestamps and a cutoff exactly on a token's time →
  asserts that token **is** included (inclusive `<=`).
- A timeline longer than `max_len` with a late cutoff → asserts the kept tokens are
  the **last** `max_len` before the cutoff, not the first (§3.2).
- A cutoff before the first event → `n_past == 0`, row removed, `prediction_id`
  recorded in the audit output.
- One subject, three cutoffs → three output rows, correct nesting, `cut_index` row
  order matching the inference table exactly.
- `s_elapsed_past[0]` after left-trimming → asserts it is **not** zero, i.e. elapsed
  time is still measured from the subject's own first token (§3.5).

**Audit tests** — each assertion in §9 gets a crafted violation proving it fires.

**Integration smoke test** — `max_encounters_per_task: 200` on MIMIC, end to end, on
CPU/MPS. Asserts the pipeline completes and every task produces a valid FLAIR
report bundle. Metrics are not asserted.

---

## 13. Compute

The model is 76.9M parameters, inference only. `cotorra` selects MPS when CUDA is
absent (`extractor.py:44-50`), so a subsampled run is feasible on the development
Mac; the slow stages there are stage 0 (clifpy over ~430 MB of labs) and stage 2
(collation), not the forward passes.

A full run over four tasks is HPC work. The two daily tasks produce one row per
07:00 window per stay, so the deduplicated union is estimated in the high hundreds
of thousands of rows at up to 4096 tokens each — order a few GPU-hours on one A100.
`PLAN.md` carries the runbook and the `verify-rope` gate that runs first.

---

## 14. Known limitations

Stated in `PLAN.md` and in the report bundle's accompanying notes.

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

---

## 15. Open items

| Item | Resolution path |
|---|---|
| `time_based_rope` setting and `sec_per_pos_id` | `gemflair verify-rope` against the recorded 2.2565 eval loss; confirm with the weights' authors if the run is ambiguous |
| Whether `clifpy` reproduces the exact derived tables used for GEM training | Compare `UNK` rate and vocabulary coverage after stage 3; a low `UNK` rate is evidence the derivation matches |
| True cohort sizes for the two daily tasks | Measured at stage 1 on the first run; determines whether `max_encounters_per_task` is needed on HPC |
