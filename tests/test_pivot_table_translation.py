# tests/test_pivot_table_translation.py
import pytest
from pd2pl import translate_code

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic pivot_table (implicit aggfunc='mean')
        (
            "df.pivot_table(index='idx', columns='col', values='val')",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='mean')"
        ),
        # Explicit aggfunc='sum'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='sum')",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='sum')"
        ),
        # Explicit aggfunc='count'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='count')",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='count')"
        ),
        # Explicit aggfunc='max'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='max')",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='max')"
        ),
         # Explicit aggfunc='median'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='median')",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='median')"
        ),
        # Pivot_table with list values and default aggfunc
         (
            "df.pivot_table(index='idx', columns='col', values=['v1', 'v2'])",
            "df_pl.pivot(index='idx', columns='col', values=['v1', 'v2'], aggregate_function='mean')"
        ),
        # Pivot_table with fill_value (implicit aggfunc='mean')
        (
            "df.pivot_table(index='idx', columns='col', values='val', fill_value=0)",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='mean').fill_null(0)"
        ),
        # Pivot_table with fill_value and explicit aggfunc='sum'
        (
            "df.pivot_table(index='idx', columns='col', values='val', aggfunc='sum', fill_value=-1)",
            "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='sum').fill_null(-1)"
        ),
        # List index/columns
        (
            "df.pivot_table(index=['i1'], columns=['c1', 'c2'], values='val')",
            "df_pl.pivot(index=['i1'], columns=['c1', 'c2'], values='val', aggregate_function='mean')"
        ),
        # Test ignoring sort parameter
        (
            "df.pivot_table(index='idx', columns='col', values='val', sort=False)",
             "df_pl.pivot(index='idx', columns='col', values='val', aggregate_function='mean')"
        ),
    ]
)
def test_pivot_table_translations(pandas_code, expected_polars):
    """Test the translation of pandas pivot_table to polars pivot."""
    assert translate_code(pandas_code.strip()) == expected_polars.strip()

# TODO: Add tests for unsupported arguments (margins, dropna, list/dict aggfunc)
# These should ideally check for NotImplementedError or warnings once implemented. 