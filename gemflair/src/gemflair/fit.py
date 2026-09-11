import numpy as np
import polars as pl
import xgboost as xgb


def attach(cohort, index, features):
    df = cohort.join(index, left_on=["hospitalization_id", "feature_cutoff_dttm"],
                     right_on=["subject_id", "feature_cutoff_dttm"], how="inner")
    # Joins do not preserve row order; sort so row_ix indexing lines up with df.
    df = df.sort("row_ix")
    X = np.stack(features["features"].to_numpy())[df["row_ix"].to_numpy()]
    return df, X.astype(np.float32)


def val_mask(df, train, seed):
    enc = df.filter(pl.Series(train))["hospitalization_join_id"].unique().sort()
    rng = np.random.default_rng(seed)
    held = set(rng.choice(enc.to_numpy(), max(1, round(0.1 * len(enc))), replace=False))
    return np.array([e in held for e in df["hospitalization_join_id"]])


def run(cohort, index, features, label, cfg):
    df, X = attach(cohort, index, features)
    y = df[label].to_numpy()
    train = (df["split"] == "train").to_numpy()
    val = val_mask(df, train, cfg["xgboost"]["seed"]) & train
    model = xgb.XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
        random_state=cfg["xgboost"]["seed"])
    model.fit(X[train & ~val], y[train & ~val],
              eval_set=[(X[val], y[val])], verbose=False)
    return df.with_columns(y_prob=pl.Series(model.predict_proba(X)[:, 1]))
