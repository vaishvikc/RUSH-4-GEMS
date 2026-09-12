import pathlib
from datetime import datetime
from types import SimpleNamespace

import clifpy
import polars as pl
import pytest

from gemflair import prep

DEMO = pathlib.Path(clifpy.__file__).parent / "data" / "clif_demo"


def test_tables_cover_the_collation_requirements():
    assert set(prep.TABLES) == {
        "clif_medication_admin_continuous_converted",
        "clif_medication_admin_intermittent_converted",
        "clif_respiratory_support_processed",
        "clif_sofa",
    }


def test_missing_table_error_names_the_token_prefixes():
    with pytest.raises(RuntimeError) as e:
        prep.fail_missing("clif_sofa")
    assert "SOFA" in str(e.value)


def test_check_rejects_a_schema_valid_frame_with_no_rows():
    time_col, cols = prep.REQUIRED["clif_sofa"]
    schema = {time_col: pl.Datetime, "hospitalization_id": pl.String}
    schema.update({c: pl.Int32 for c in cols if c not in schema})
    frame = pl.DataFrame(schema=schema)
    assert set(frame.columns) == set([time_col] + cols) and frame.is_empty()
    with pytest.raises(RuntimeError, match="no rows"):
        prep.check(frame, "clif_sofa")


def test_check_rejects_a_frame_whose_time_column_was_renamed():
    frame = pl.DataFrame({c: [0] for c in ["recorded_dttm"] + prep.REQUIRED["clif_sofa"][1]})
    with pytest.raises(RuntimeError, match="event_time"):
        prep.check(frame, "clif_sofa")


def test_check_rejects_nulls_in_the_time_column():
    time_col, cols = prep.REQUIRED["clif_respiratory_support_processed"]
    frame = pl.DataFrame({c: [None] if c == time_col else [0] for c in [time_col] + cols})
    with pytest.raises(RuntimeError, match="RESP"):
        prep.check(frame, "clif_respiratory_support_processed")


def test_sofa_excludes_active_admissions(monkeypatch):
    hospitalizations = pl.DataFrame({
        "hospitalization_id": ["complete", "active"],
        "admission_dttm": [datetime(2025, 1, 1), datetime(2026, 1, 1)],
        "discharge_dttm": [datetime(2025, 1, 2), None],
    }).to_pandas()
    co = SimpleNamespace(
        data_directory="/data", filetype="parquet", timezone="US/Central",
        load_table=lambda name: SimpleNamespace(df=hospitalizations))

    def compute_sofa(data_directory, stays, **kwargs):
        assert stays["hospitalization_id"].to_list() == ["complete"]
        return pl.DataFrame({"hospitalization_id": ["complete"]})

    monkeypatch.setattr(clifpy, "compute_sofa_polars", compute_sofa)
    result = prep._build_one(co, "clif_sofa")
    assert result["hospitalization_id"].to_list() == ["complete"]
    assert result["event_time"].null_count() == 0


def test_build_on_clifpy_demo_data_matches_the_collation_columns(tmp_path):
    cfg = {"data": {"clif_dir": DEMO, "derived_dir": tmp_path, "timezone": "UTC"}}
    for path in prep.build(cfg):
        time_col, cols = prep.REQUIRED[path.stem]
        frame = pl.read_parquet(path)
        assert not frame.is_empty()
        assert set([time_col] + cols) <= set(frame.columns)
