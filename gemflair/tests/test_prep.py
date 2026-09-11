import pytest

from gemflair import prep


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
