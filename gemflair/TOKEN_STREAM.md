# Token Stream

## Overview

`gemflair` turns longitudinal EHR events into prediction-time patient states and uses those states for clinical decision support (CDS) tasks.

The central separation is:

-   The **token stream** contains only timestamped clinical history available at a prediction cutoff.
-   The **task branch** defines prediction cutoffs, train/test splits, and labels.
-   The frozen GEM receives tokens and elapsed times, but never labels.
-   A separate XGBoost classifier is trained for each downstream task.

This document records the architecture and the implementation decisions needed to run the full RUSH evaluation with the FedAvg10 GEM checkpoint.

## Architecture

``` mermaid
flowchart TD
    EHR["Source EHR systems"] --> EXTRACT["CLIF extract / ETL"]
    EXTRACT --> RAW["Raw CLIF parquet tables"]

    subgraph TOKEN["Token stream branch"]
        RAW --> CLIFPY["CLIFPy derivation"]
        CLIFPY --> DERIVED["Derived clinical tables<br/>medications, respiratory support, SOFA"]
        RAW --> VIEW["Unified raw + derived CLIF view"]
        DERIVED --> VIEW
        VIEW --> COLLATE["DST-safe Cocoa collation<br/>normalize code, value, and event time"]
        COLLATE --> TOKENIZER["Frozen pretrained tokenizer<br/>fixed vocabulary and numeric bins"]
        TOKENIZER --> STREAM["Chronological token stream<br/>per hospitalization"]
    end

    subgraph TASKS["Task and label branch"]
        RAW --> FLAIR["FLAIR cohort construction"]
        FLAIR --> CUTOFFS["Prediction rows<br/>label + split + prediction cutoff"]
    end

    STREAM --> WINNOW["Leakage-safe winnower<br/>event time <= prediction cutoff"]
    CUTOFFS --> WINNOW
    WINNOW --> CONTEXT["Cutoff-safe context<br/>most recent 4,096 tokens"]
    WINNOW --> INDEX["Cut index<br/>identity + cutoff + row order"]
    WINNOW --> AUDIT["Leakage and integrity audit"]

    CONTEXT --> SHARD["Contiguous multi-GPU sharding"]
    SHARD --> GEM["Frozen FedAvg10 GEM<br/>inference only, no labels"]
    GEM --> MERGE["Ordered merge by row index<br/>coverage and dimension checks"]
    INDEX --> MERGE
    MERGE --> STATE["Patient state at prediction time<br/>1,024-dimensional vector"]

    STATE --> MORT["XGBoost<br/>ICU daily mortality"]
    STATE --> LTACH["XGBoost<br/>ICU daily LTACH"]
    STATE --> EXTUB["XGBoost<br/>extubation failure in 24 h"]
    STATE --> READMIT["XGBoost<br/>ICU readmission"]

    CUTOFFS -.->|task labels and splits| MORT
    CUTOFFS -.->|task labels and splits| LTACH
    CUTOFFS -.->|task labels and splits| EXTUB
    CUTOFFS -.->|task labels and splits| READMIT

    MORT --> PREDS["Task prediction parquet files"]
    LTACH --> PREDS
    EXTUB --> PREDS
    READMIT --> PREDS
    PREDS --> REPORTS["FLAIR test reports<br/>metrics + visualizations"]

    classDef source fill:#e8eef7,stroke:#355070,color:#172033;
    classDef frozen fill:#dff3e4,stroke:#2d6a4f,color:#15331f;
    classDef trained fill:#fff0d5,stroke:#bc6c25,color:#4d2a0a;
    classDef safety fill:#f5e5f5,stroke:#7b2c83,color:#35123a;
    classDef output fill:#e6f4f8,stroke:#247b9e,color:#123845;

    class EHR,EXTRACT,RAW,DERIVED,VIEW source;
    class TOKENIZER,GEM frozen;
    class MORT,LTACH,EXTUB,READMIT trained;
    class WINNOW,AUDIT,INDEX,MERGE safety;
    class STREAM,CONTEXT,STATE,PREDS,REPORTS output;
```

### How to read the diagram

The EHR is first standardized into CLIF. The same CLIF source then feeds two independent branches:

1.  The token branch constructs a chronological clinical timeline.
2.  The task branch constructs labels and prediction cutoffs.

The branches meet only at the winnower. For every prediction cutoff, the winnower selects events at or before that cutoff and removes future events. The frozen GEM compresses that context into one patient-state vector. Four independent XGBoost models then learn task-specific mappings from those vectors to outcome probabilities.

The patient state is not a permanent embedding for a person. It describes one hospitalization at one prediction time. A hospitalization with multiple daily prediction windows therefore has multiple patient states.

## Token Flow

The data transformation can be summarized as:

``` text
timestamped EHR events
  -> CLIF tables
  -> normalized clinical events
  -> pretrained token IDs and elapsed times
  -> prediction-time context
  -> frozen GEM patient state
  -> task-specific CDS probability
```

### 1. EHR to CLIF

The source EHR is represented as CLIF parquet tables. CLIF gives downstream tools stable table, identifier, timestamp, code, and value conventions.

For this run, the CLIF source was:

``` text
/home/vchaudha/RUSH_CLIF
```

### 2. Derived clinical tables

CLIFPy creates the extra tables expected by the pretrained tokenizer:

| Derived table | Token family | Important decision |
|------------------------|------------------------|------------------------|
| Continuous medication administrations | `MED-CTS` | Preserve the pretrained medication units. |
| Intermittent medication administrations | `MED-INT` | Preserve the pretrained medication units. |
| Processed respiratory support | `RESP` | Use the CLIFPy respiratory waterfall. |
| Stay-level SOFA | `SOFA` | Timestamp at discharge and exclude active admissions from this terminal table. |

Medication units and numeric bins are not relearned at RUSH. Changing them would silently change token identities relative to GEM pretraining.

### 3. Collation and tokenization

Cocoa collates the raw and derived tables into a single event stream for each hospitalization. Events are normalized to a subject, event time, code, and optional numeric or text value.

The tokenizer is frozen and loaded from the pretrained GEM artifacts. It:

-   Uses the original vocabulary and numeric bins.
-   Adds the expected sequence boundary tokens.
-   Orders simultaneous events deterministically.
-   Maps unseen events to `UNK` instead of expanding the vocabulary.
-   Produces token IDs together with event times.

The result is `processed/tokens_times.parquet`, with one ordered timeline per hospitalization.

### 4. Prediction-time winnowing

FLAIR creates task rows containing a `feature_cutoff_dttm`. The winnower unions and deduplicates all task cutoffs, then applies the rule:

``` text
include event when event_time <= feature_cutoff_dttm
```

The cutoff is inclusive. Events after it cannot reach GEM. Cut points with no prior events are removed and reported by the audit.

If a context exceeds 4,096 tokens, only the **most recent** 4,096 are retained. Elapsed time remains relative to the original hospitalization timeline; it is not reset after truncation.

The winnower writes:

| Artifact | Purpose |
|------------------------------------|------------------------------------|
| `held_out_for_inference.parquet` | Cutoff-safe tokens and elapsed times for GEM. |
| `cut_index.parquet` | Patient identity, cutoff, lengths, and authoritative row order. |
| `cut_index.fingerprint` | Detects a changed cutoff index before fitting. |
| `audit/winnow.json` | Leakage, duplicate, truncation, removal, and `UNK` checks. |

`held_out` is a Cotorra filename convention. It contains both FLAIR train and test cut points. The statistical split remains in each task cohort and is enforced during XGBoost fitting.

### 5. Frozen GEM patient states

Cotorra loads the pretrained decoder-only GEM in evaluation and inference mode. No optimizer, backward pass, parameter update, or task label is involved.

For each cutoff-safe sequence, GEM emits a 1,024-dimensional final hidden state. The feature width is also validated against the checkpoint's `hidden_size`, so a changed checkpoint cannot silently produce an unexpected representation size.

The final feature file is:

``` text
processed/features-held_out-coreopsis-round-10.parquet
```

### 6. Task-specific CDS models

There is no single XGBoost model for the entire dataset. Four classifiers are trained independently:

| Task model                         | Prediction unit              |
|------------------------------------|------------------------------|
| ICU daily mortality                | Daily ICU prediction window. |
| ICU daily LTACH                    | Daily ICU prediction window. |
| Extubation failure within 24 hours | Eligible extubation episode. |
| ICU readmission                    | Eligible ICU stay.           |

Each classifier uses only its task's FLAIR training rows. A seeded 10% validation set is selected from whole stitched encounters inside the training split for early stopping. Test encounters are not used for fitting or early stopping.

The classifier generates probabilities for all attached rows, and FLAIR computes reported metrics from the test split only.

## Run Decisions

The following settings describe the completed RUSH/FedAvg10 run.

| Area | Decision | Reason |
|------------------------|------------------------|------------------------|
| Site | `rush` | Select the RUSH CLIF dataset and FLAIR site behavior. |
| Timezone | `US/Central` | Match the source site's local clinical timestamps. |
| GEM checkpoint | `FedAvg10/coreopsis-round-10` | Evaluate the selected federated pretrained checkpoint. |
| GEM training | Frozen | Evaluate representations without task fine-tuning. |
| Tokenizer | Frozen pretrained tokenizer | Preserve the vocabulary and bins used during GEM pretraining. |
| Context limit | 4,096 tokens | Respect the checkpoint position limit and bound extraction memory. |
| Long context policy | Keep latest 4,096 | Preserve history closest to the prediction time. |
| Cutoff policy | Inclusive, `event_time <= cutoff` | Admit information available at the prediction instant while blocking future data. |
| Positional encoding | Plain positions | Lower verification loss than the tested time-based scheme. |
| Extraction batch size | 64 | Fit the L40S memory envelope while maintaining throughput. |
| GPUs | L40S devices 0 and 1 | Parallelize the only GPU-bound stage. |
| GPU partitioning | Two contiguous shards | Preserve deterministic positional alignment with the cut index. |
| XGBoost models | One per task | Labels and prediction units differ across the four CDS tasks. |
| XGBoost seed | 42 | Reproducible validation sampling and model fitting. |
| Early stopping | 50 rounds | Stop task training when validation loss no longer improves. |
| Encounter cap | None | Run all available eligible encounters. |
| Work directory | `work/fedavg10-rush` | Keep generated data, caches, logs, and reports outside source directories. |

## RoPE Decision Gate

The checkpoint does not publish whether training used time-based rotary position IDs. Choosing incorrectly does not crash extraction; it silently changes every patient state. The pipeline therefore includes an empirical gate after winnowing and before full extraction.

The verifier compares next-token loss on a deterministic tuning subset:

| Position scheme |        Measured loss |
|-----------------|---------------------:|
| Plain positions |  `2.540255144238472` |
| Time-based RoPE | `2.6295342594385147` |

Plain positions had the lower loss, so the completed configuration uses:

``` yaml
model:
  time_based_rope: null
```

This is a human-reviewed gate. `gemflair run` intentionally does not run or apply `verify-rope` automatically.

## Workarounds and Safeguards

### Timestamp and DST handling

RUSH timestamps exposed three separate compatibility concerns:

| Concern | Workaround |
|------------------------------------|------------------------------------|
| Timezone consistency | Propagate `US/Central` to generated FLAIR and Cocoa configuration. |
| Nonexistent spring-forward times | Retry naive nonexistent timestamps one hour later before localization. |
| Ambiguous fall-back times | Select the later local instant consistently. |
| Nanosecond vs microsecond cutoffs | Cast cohort cutoff precision to the cut-index precision before joins. |

The precision normalization was required when `icu_readmission` used `datetime[ns]` while the cut index used `datetime[us]`. Without normalization, the first three task models completed but readmission fitting stopped at the Polars join. The corrected fit path is covered by a regression test.

### Terminal SOFA and active admissions

CLIFPy's stay-level SOFA output does not provide an event timestamp. The pipeline uses `discharge_dttm` as its event time. Active admissions have no discharge time, so they are excluded from terminal SOFA derivation rather than assigning an invented timestamp. Their other raw CLIF events remain available.

Because terminal SOFA is stamped at discharge, it is normally excluded from earlier prediction contexts by the winnower. This prevents leakage but means these SOFA tokens rarely contribute to the current tasks.

### Multi-GPU extraction

Cotorra's built-in sharding was not used because interleaved rows can break the positional relationship between features and `cut_index.parquet`.

The implemented extraction instead:

1.  Splits the inference rows into contiguous ranges.
2.  Loads one complete frozen GEM copy on each GPU.
3.  Carries the global `_row_ix` through each worker.
4.  Sorts the merged output by `_row_ix`.
5.  Verifies every expected row appears exactly once.
6.  Verifies every patient state has the checkpoint's hidden size.
7.  Atomically replaces the final feature parquet.

For the completed run, the two shards contained 132,141 and 132,140 rows.

### Runtime compatibility

Native Torch Triton overrides are disabled before RoPE verification and extraction. This avoids a runtime incompatibility in the installed model stack without changing GEM weights or inference semantics.

### Work and cache isolation

All generated artifacts are rooted under:

``` text
/home/vchaudha/RUSH-4-GEMS/work/fedavg10-rush
```

The detached run also redirected the Hugging Face cache, generic cache, and temporary directory into that tree. Generated work directories are ignored by Git.

### Audits and stale-feature protection

The pipeline fails early on incompatible vocabulary size, boundary token IDs, maximum sequence length, or timezone configuration.

After winnowing it checks:

-   No included event occurs after its prediction cutoff.
-   No eligible event at or before the cutoff was skipped.
-   Token and elapsed-time arrays have matching lengths.
-   Contexts do not exceed 4,096 tokens.
-   Deduplicated cut points are unique.
-   Removed rows, truncation, stitched encounters, and `UNK` usage are recorded.

After extraction it checks that exactly one final feature file exists and that its row count matches the cut index. Fitting also compares cutoff fingerprints so that a new winnow result is not accidentally joined to stale features.

## Operational Sequence

Run commands from the `gemflair/` directory because relative checkpoint and tokenizer paths resolve against the process working directory.

``` bash
uv run gemflair prep        -c config/gemflair.yaml
uv run gemflair cohorts     -c config/gemflair.yaml
uv run gemflair tokenize    -c config/gemflair.yaml
uv run gemflair winnow      -c config/gemflair.yaml
uv run gemflair verify-rope -c config/gemflair.yaml
# Review audit/verify_rope.json and update the configuration if needed.
uv run gemflair extract     -c config/gemflair.yaml
uv run gemflair fit         -c config/gemflair.yaml
uv run gemflair report      -c config/gemflair.yaml
```

Only `extract` requires GPUs. Preparation, cohort construction, collation, tokenization, winnowing, XGBoost fitting, and report generation are CPU and I/O stages.

Long-running stages can be detached, but the working directory must remain `gemflair/`. Starting a detached command from the repository root causes relative model paths to resolve incorrectly.

Preparation and cohort generation reuse existing outputs. Extraction shard files are temporary and are removed after a successful ordered merge.

## Completed Run

### Feature extraction

| Measurement                         |  Result |
|-------------------------------------|--------:|
| Unique prediction cut points        | 264,281 |
| Final patient states                | 264,281 |
| Patient-state width                 |   1,024 |
| Contexts truncated to 4,096 tokens  |  64,461 |
| Global `UNK` rate                   |  21.86% |
| Missing or duplicate extracted rows |       0 |

### Downstream predictions

| Task                               | Prediction rows | Null probabilities |
|------------------------------------|----------------:|-------------------:|
| ICU daily mortality                |         227,222 |                  0 |
| ICU daily LTACH                    |         227,222 |                  0 |
| Extubation failure within 24 hours |           7,311 |                  0 |
| ICU readmission                    |          29,755 |                  0 |

All four report bundles completed. The final test suite contained 83 passing tests.

The report tool emitted a non-fatal temporal split warning for mortality and LTACH: 585 of 171,227 training rows, approximately 0.3%, occur at or after the reported test start, with overlap reaching 126 days. The reports were generated, but this cohort split warning should be considered when interpreting results.

## Output Map

All paths below are relative to `work/fedavg10-rush/`.

| Path | Contents |
|------------------------------------|------------------------------------|
| `clif_derived/` | CLIFPy-derived model input tables. |
| `raw_view/` | Unified symlink view of raw and derived CLIF parquet files. |
| `cohorts/` | Four FLAIR task cohorts and cohort summaries. |
| `processed/tokens_times.parquet` | Hospitalization token streams and event times. |
| `processed/held_out_for_inference.parquet` | Cutoff-safe GEM inputs. |
| `processed/cut_index.parquet` | Identity and row-order contract for patient states. |
| `processed/features-held_out-coreopsis-round-10.parquet` | Frozen GEM patient states. |
| `preds/` | Four task prediction parquet files. |
| `reports/` | Four report bundles with JSON metrics and PNG visualizations. |
| `audit/` | Startup, winnow, RoPE, and extraction checks. |
| `logs/` | Detached pipeline and recovery logs. |
| `cache/`, `tmp/` | Run-local caches and temporary files. |

## Known Limitations

1.  **Long histories are truncated.** Contexts longer than 4,096 tokens lose older history, although the original elapsed-time origin is retained.
2.  **Vocabulary transfer is imperfect.** RUSH events absent from the pretrained vocabulary become `UNK`; the measured global rate was 21.86%.
3.  **Context is hospitalization-level.** FLAIR can stitch several hospitalizations into one encounter, but GEM processes one hospitalization timeline at a time.
4.  **Zero-history prediction rows are removed.** A task row with no event at or before its cutoff cannot receive a patient state.
5.  **Terminal SOFA is rarely usable.** It is timestamped at discharge and therefore normally falls after task prediction cutoffs.
6.  **RoPE selection is procedural.** The verifier informs a human decision; it does not automatically change configuration or block extraction.
7.  **The cutoff fingerprint is narrow.** It protects subject/cutoff row alignment but does not fingerprint token contents, elapsed times, model identity, or feature bytes.
8.  **XGBoost models are not serialized.** The current implementation saves prediction parquet files, not the fitted classifier objects. Re-running `fit` retrains them.
9.  **Temporal split overlap needs review.** The mortality and LTACH report warning is non-fatal but may affect interpretation of benchmark estimates.

## Implementation References

| Concern | Primary implementation |
|------------------------------------|------------------------------------|
| Configuration and generated paths | `config/gemflair.yaml`, `src/gemflair/config.py` |
| CLIFPy-derived tables | `src/gemflair/prep.py` |
| Collation, tokenization, GPU sharding, and reporting | `src/gemflair/run.py` |
| Cutoff-safe token selection | `src/gemflair/winnow.py` |
| Leakage and integrity checks | `src/gemflair/audit.py` |
| Plain vs time-based position verification | `src/gemflair/verify_rope.py` |
| Task-specific XGBoost fitting | `src/gemflair/fit.py` |
| Command orchestration | `src/gemflair/cli.py` |
| Detailed runbook | `PLAN.md` |
