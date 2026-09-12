import json

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForCausalLM

from gemflair import config


def disable_native_triton():
    try:
        from torch._native import triton_utils
    except ImportError:
        return
    triton_utils.deregister_op_overrides()


def training_target(model_cfg):
    state = model_cfg.get("trainer_state")
    return json.loads(state.read_text())["best_metric"] if state else None


def blocks(rows, seq_len):
    flat = [x for row in rows for x in row]
    n = len(flat) // seq_len
    return np.array(flat[: n * seq_len], dtype=np.int64).reshape(n, seq_len)


def eval_loss(model, tok_blocks, elapsed_blocks, sec_per_pos_id, device, batch_size=4):
    if len(tok_blocks) == 0:
        raise ValueError("no blocks to evaluate - check for an empty tuning split")
    losses = []
    for i in range(0, len(tok_blocks), batch_size):
        ids = torch.tensor(tok_blocks[i:i + batch_size], device=device)
        pos = None
        if sec_per_pos_id:
            e = torch.tensor(elapsed_blocks[i:i + batch_size], device=device,
                             dtype=torch.float32) / sec_per_pos_id
            pos = (e + torch.arange(ids.shape[-1], device=device)).long()
        with torch.inference_mode():
            losses.append(model(input_ids=ids, position_ids=pos, labels=ids).loss.item())
    return float(np.mean(losses))


def run(cfg, n_blocks=64):
    p = config.paths(cfg)
    target = training_target(cfg["model"])
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        disable_native_triton()
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["dir"]).to(device).eval()

    splits = pl.read_parquet(p["processed"] / "subject_splits.parquet")
    tt = (pl.read_parquet(p["processed"] / "tokens_times.parquet")
          .join(splits.filter(pl.col("split") == "tuning"), on="subject_id")
          .with_columns(s_elapsed=pl.col("times").list.eval(
              (pl.element() - pl.element().first()).dt.total_seconds())))

    n = cfg["model"]["max_len"]
    tok = blocks(tt["tokens"].to_list(), n)[:n_blocks]
    ela = blocks(tt["s_elapsed"].to_list(), n)[:n_blocks]
    sec = (cfg["model"]["time_based_rope"] or {}).get("sec_per_pos_id", 300)

    with_rope = eval_loss(model, tok, ela, sec, device)
    without = eval_loss(model, tok, ela, None, device)
    return {"target": target, "with_rope": with_rope, "without_rope": without,
            "selected": "time_based_rope" if with_rope < without else "plain"}
