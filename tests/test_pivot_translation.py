# tests/test_pivot_translation.py
import pytest
import pandas as pd
import polars as pl
from pd2pl import translate_code
from tests._helpers import compare_frames

@pytest.fixture
def long_df_unique():
    """Fixture for pivot tests (unique index/column combinations)."""
    return pd.DataFrame({
        'idx': ['R1', 'R1', 'R2', 'R2'],
        'col': ['C1', 'C2', 'C1', 'C2'],
        'val': [10, 20, 30, 40],
        'v1': [1, 2, 3, 4],
        'v2': [5, 6, 7, 8]
    })

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic pivot with single index, columns, values (Uses correct names: idx, col, val)
        (
            "df.pivot(index='idx', columns='col', values='val')",
            "df_pl.pivot(index='idx', on='col', values='val')"
        ),
        # List index, single columns, single values (FIXED: Use ['idx'] instead of ['i1', 'i2'])
        (
            "df.pivot(index=['idx'], columns='col', values='val')",
            "df_pl.pivot(index=['idx'], on='col', values='val')"
        ),
        # Single index, list columns, single values (FIXED: Use ['col'] instead of ['c1', 'c2'])
        (
            "df.pivot(index='idx', columns=['col'], values='val')",
            "df_pl.pivot(index='idx', on=['col'], values='val')"
        ),
        # Single index, single columns, list values (Uses correct names: idx, col, ['v1', 'v2'])
        (
            "df.pivot(index='idx', columns='col', values=['v1', 'v2'])",
            "df_pl.pivot(index='idx', on='col', values=['v1', 'v2'])"
        ),
        # All list arguments (FIXED: Use ['idx'], ['col'], ['v1', 'v2'])
        (
            "df.pivot(index=['idx'], columns=['col'], values=['v1', 'v2'])",
            "df_pl.pivot(index=['idx'], on=['col'], values=['v1', 'v2'])"
        ),
        # Test a different argument order (Uses correct names: val, col, idx)
        (
            "df.pivot(values='val', columns='col', index='idx')",
            "df_pl.pivot(values='val', on='col', index='idx')"
        ),
        # Case added during previous planning (already uses correct names)
        (
            "df.pivot(index=['idx'], columns=['col'], values=['val'])",
            "df_pl.pivot(index=['idx'], on=['col'], values=['val'])"
        ),
    ]
)
def test_pivot_translations(pandas_code, expected_polars, long_df_unique):
    """Test the translation and functional equivalence of pandas pivot to polars pivot."""
    # Note: This assumes 'values' is always explicitly provided in pandas call.
    # Polars uses aggregate_function='first' by default.
    translated_polars = translate_code(pandas_code.strip())
    assert translated_polars == expected_polars.strip()
    # Add functional comparison
    assert compare_frames(pandas_code.strip(), translated_polars, long_df_unique), \
           f"DataFrame comparison failed for:\nPandas: {pandas_code}\nPolars: {translated_polars}" 