# gemflair

Evaluates a pretrained GEM as a frozen feature extractor on the FLAIR ICU benchmark.

```bash
cd gemflair
uv sync                                    # install (wheels are vendored)
$EDITOR config/gemflair.yaml               # set data.clif_dir, model.dir, work_dir
uv run gemflair run -c config/gemflair.yaml
```

The RUSH/FedAvg10 configuration writes all generated artifacts beneath
`../work/fedavg10-rush` and extracts on GPUs 0 and 1.

`uv run gemflair --help` lists the stages; each can be run alone. Run
`uv run gemflair verify-rope` after `winnow` and **before** `extract` — see `PLAN.md`,
which carries the run order, the HPC runbook, and the known limitations.
