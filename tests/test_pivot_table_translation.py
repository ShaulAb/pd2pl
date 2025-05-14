# tests/test_pivot_table_translation.py
import pytest
import pandas as pd
import polars as pl

from tests._helpers import compare_frames
from tests.conftest import translate_test_code

@pytest.fixture
def long_df_duplicates():
    """Fixture for pivot_table tests (duplicate index/column combinations)."""
    return pd.DataFrame({
        'idx': ['R1', 'R1', 'R2', 'R2', 'R1', 'R2'], # R1/C1 repeats, R2/C2 repeats
        'col': ['C1', 'C2', 'C1', 'C2', 'C1', 'C2'],
        'val': [1, 2, 3, 4, 5, 6], # Values for aggregation
        'v1': [10,20,30,40,50,60],
        'v2': [11,21,31,41,51,61]
    })

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic pivot_table (implicit aggfunc='mean')
        (
            "df.pivot_table(index='idx', columns='col', values='val')",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='mean')"
        ),
        # Explicit aggfunc='sum'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='sum')",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='sum')"
        ),
        # Explicit aggfunc='count'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='count')",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='len')"
        ),
        # Explicit aggfunc='max'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='max')",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='max')"
        ),
         # Explicit aggfunc='median'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='median')",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='median')"
        ),
        # Pivot_table with list values and default aggfunc
         (
            "df.pivot_table(index='idx', columns='col', values=['v1', 'v2'])",
            "df_pl.pivot(index='idx', on='col', values=['v1', 'v2'], aggregate_function='mean')"
        ),
        # Pivot_table with fill_value (implicit aggfunc='mean')
        (
            "df.pivot_table(index='idx', columns='col', values='val', fill_value=0)",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='mean').fill_null(0)"
        ),
        # Pivot_table with fill_value and explicit aggfunc='sum'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='sum', fill_value=-1)",
            "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='sum').fill_null(-1)"
        ),
        # List index/columns
        (
            "df.pivot_table(index=['idx'], columns=['col'], values='val')",
            "df_pl.pivot(index=['idx'], on=['col'], values='val', aggregate_function='mean')"
        ),
        # Test ignoring sort parameter
        (
            "df.pivot_table(index='idx', columns='col', values='val', sort=False)",
             "df_pl.pivot(index='idx', on='col', values='val', aggregate_function='mean')"
        ),
    ]
)
def test_pivot_table_translations(pandas_code, expected_polars, long_df_duplicates):
    """Test the translation and functional equivalence of pandas pivot_table to polars pivot."""
    translated_polars = translate_test_code(pandas_code.strip())
    assert translated_polars == expected_polars.strip()
    assert compare_frames(pandas_code.strip(), translated_polars, long_df_duplicates), \
           f"DataFrame comparison failed for:\nPandas: {pandas_code}\nPolars: {translated_polars}"

# TODO: Add tests for unsupported arguments (margins, dropna, list/dict aggfunc)
# These should ideally check for NotImplementedError or warnings once implemented. 