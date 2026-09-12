import json
import multiprocessing
import pathlib
import subprocess

import numpy as np
import polars as pl
from cocoa.collator import Collator

from gemflair import config


class _DstSafeCollator(Collator):
    def to_default_tz(self, df, column):
        if self.is_tz_aware(df.collect_schema(), column):
            return pl.col(column).dt.convert_time_zone(self.tz)
        value = pl.col(column).cast(pl.Datetime)
        localized = value.dt.replace_time_zone(
            self.tz, ambiguous="latest", non_existent="null")
        shifted = (value + pl.duration(hours=1)).dt.replace_time_zone(
            self.tz, ambiguous="latest", non_existent="null")
        return pl.coalesce(localized, shifted)


def _sh(*args):
    subprocess.run([str(a) for a in args], check=True)


def raw_view(cfg):
    # cocoa takes one --raw-data-home, so link both directories into one view.
    view = config.paths(cfg)["raw_view"]
    view.mkdir(parents=True, exist_ok=True)
    for src in (cfg["data"]["clif_dir"], cfg["data"]["derived_dir"]):
        for f in sorted(src.glob("*.parquet")) if src.is_dir() else []:
            link = view / f.name
            link.unlink(missing_ok=True)
            link.symlink_to(f)
    return view


def subsample(cohort, n, seed):
    if n is None:
        return cohort
    rng = np.random.default_rng(seed)
    keep = []
    for split in sorted(cohort["split"].unique()):
        enc = cohort.filter(pl.col("split") == split)["hospitalization_join_id"].unique().sort()
        keep += list(rng.choice(enc.to_numpy(), min(n, len(enc)), replace=False))
    return cohort.filter(pl.col("hospitalization_join_id").is_in(keep))


def load_task(cfg, task, force=False):
    p = config.paths(cfg)
    out = p["cohorts"] / f"{task}.parquet"
    # A cohort takes ~77 minutes to build; skip it on a resumed run.
    if not out.exists() or force:
        args = ["flair", "load-task", task,
                "--clif-config", p["generated"] / "clif_config.json", "--out", out]
        if cfg["split"]["train_end"] and cfg["split"]["test_start"]:
            args += ["--train-end", cfg["split"]["train_end"],
                     "--test-start", cfg["split"]["test_start"]]
        _sh(*args)
    trimmed = subsample(pl.read_parquet(out),
                        cfg["limits"]["max_encounters_per_task"], cfg["xgboost"]["seed"])
    trimmed.write_parquet(out)
    return out


def collate(cfg):
    p = config.paths(cfg)
    collator = _DstSafeCollator(
        collation_cfg=p["generated"] / "collation.yaml",
        raw_data_home=raw_view(cfg), processed_data_home=p["processed"])
    collator.save_all()


def tokenize(cfg):
    p = config.paths(cfg)
    _sh("cocoa", "tokenize", "--tokenizer-home", cfg["model"]["tokenizer"],
        "--processed-data-home", p["processed"])


def shard_bounds(total, count, index):
    size, extra = divmod(total, count)
    start = index * size + min(index, extra)
    return start, start + size + (index < extra)


def _extract_part(extraction_cfg, processed, model, part, device, count, index):
    import torch
    from cotorra.extractor import Extractor
    from gemflair.verify_rope import disable_native_triton

    torch.cuda.set_device(device)
    disable_native_triton()
    extractor = Extractor(
        extraction_cfg=extraction_cfg, processed_data_home=processed,
        model_home=model, output_home=pathlib.Path(part).parent)
    dataset = extractor.loader.for_inference["held_out"]
    start, stop = shard_bounds(len(dataset), count, index)
    shard = (dataset.select(range(start, stop))
             .add_column("_row_ix", list(range(start, stop)))
             .with_format("torch"))

    def extract_batch(batch):
        extracted = extractor.extract_final(batch)
        return {"_row_ix": extracted["_row_ix"],
                "features": extracted["features"]}

    mapped = shard.map(
        extract_batch, batched=True,
        batch_size=extractor.cfg.get("extract", {}).get("batch_size", 8),
        load_from_cache_file=False, remove_columns=shard.column_names)
    mapped.to_parquet(part)


def merge_feature_parts(parts, final_path, total, feature_size):
    frames = [pl.read_parquet(part).select("_row_ix", "features") for part in parts]
    merged = pl.concat(frames).sort("_row_ix")
    if merged["_row_ix"].to_list() != list(range(total)):
        raise RuntimeError("feature shards do not cover each row exactly once")
    lengths = merged.select(pl.col("features").list.len().unique())["features"]
    if lengths.to_list() != [feature_size]:
        raise RuntimeError(f"expected {feature_size}-value features, got {lengths.to_list()}")
    tmp = final_path.with_suffix(".tmp.parquet")
    merged.select("features").write_parquet(tmp)
    tmp.replace(final_path)
    return final_path


def extract(cfg):
    from cotorra.loader import Loader

    p = config.paths(cfg)
    extraction_cfg = p["generated"] / "extraction.yaml"
    loader = Loader(extraction_cfg, p["processed"])
    if set(loader.inference_files) != {"held_out"}:
        raise RuntimeError(f"expected only held_out inference data, got {loader.inference_files}")
    total = len(loader.for_inference["held_out"])
    devices = cfg["extract"].get("devices", [0])
    if len(devices) != len(set(devices)) or not devices:
        raise ValueError("extract.devices must contain distinct GPU indices")

    parts = [p["extract_parts"] / f"features-part-{i:02d}.parquet"
             for i in range(len(devices))]
    for part in parts:
        part.unlink(missing_ok=True)
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(
        target=_extract_part,
        args=(extraction_cfg, p["processed"], cfg["model"]["dir"], parts[i],
              device, len(devices), i))
        for i, device in enumerate(devices)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failed = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"feature extraction workers failed: {failed}")

    model_cfg = json.loads((cfg["model"]["dir"] / "config.json").read_text())
    final = p["processed"] / f"features-held_out-{cfg['model']['dir'].name}.parquet"
    merge_feature_parts(parts, final, total, model_cfg["hidden_size"])
    for part in parts:
        part.unlink()
    return [final]


def build_report(cfg, task):
    p = config.paths(cfg)
    out = p["reports"] / task
    _sh("flair", "build-report", p["preds"] / f"{task}.parquet", "--task", task,
        "--cohort", p["cohorts"] / f"{task}.parquet", "--out", out)
    return out
