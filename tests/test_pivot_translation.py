# tests/test_pivot_translation.py
import pytest
from pd2pl import translate_code

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic pivot with single index, columns, values
        (
            "df.pivot(index='idx', columns='col', values='val')",
            "df_pl.pivot(index='idx', columns='col', values='val')"
        ),
        # List index, single columns, single values
        (
            "df.pivot(index=['i1', 'i2'], columns='col', values='val')",
            "df_pl.pivot(index=['i1', 'i2'], columns='col', values='val')"
        ),
        # Single index, list columns, single values
        (
            "df.pivot(index='idx', columns=['c1', 'c2'], values='val')",
            "df_pl.pivot(index='idx', columns=['c1', 'c2'], values='val')"
        ),
        # Single index, single columns, list values
        (
            "df.pivot(index='idx', columns='col', values=['v1', 'v2'])",
            "df_pl.pivot(index='idx', columns='col', values=['v1', 'v2'])"
        ),
        # All list arguments
        (
            "df.pivot(index=['i1', 'i2'], columns=['c1', 'c2'], values=['v1', 'v2'])",
            "df_pl.pivot(index=['i1', 'i2'], columns=['c1', 'c2'], values=['v1', 'v2'])"
        ),
        # Test a different argument order in pandas call
        (
            "df.pivot(values='val', columns='col', index='idx')",
            "df_pl.pivot(values='val', columns='col', index='idx')" # Polars call order should match input
        ),
    ]
)
def test_pivot_translations(pandas_code, expected_polars):
    """Test the translation of pandas pivot to polars pivot."""
    # Note: This assumes 'values' is always explicitly provided in pandas call.
    # Polars uses aggregate_function='first' by default.
    assert translate_code(pandas_code.strip()) == expected_polars.strip() 