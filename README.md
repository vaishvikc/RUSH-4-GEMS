# RUSH-4-GEMS

This project evaluates pretrained GEM models on RUSH ICU data using the FLAIR benchmark.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
cd gemflair
uv sync
```

Update the dataset, model, and output paths in `config/gemflair.yaml`.

## Run

From the `gemflair` directory, run:

```bash
uv run gemflair run -c config/gemflair.yaml
```

For more details, see [`gemflair/README.md`](gemflair/README.md) and
[`gemflair/TOKEN_STREAM.md`](gemflair/TOKEN_STREAM.md).
