"""Test sort operation translations."""
import pytest

from tests.conftest import translate_test_code
@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic single column sort
        (
            "df.sort_values('column_a')",
            "df_pl.sort('column_a')"
        ),
        # Single column as list
        (
            "df.sort_values(['column_a'])",
            "df_pl.sort('column_a')"
        ),
        # Multiple columns
        (
            "df.sort_values(['column_a', 'column_b'])",
            "df_pl.sort(['column_a', 'column_b'])"
        ),
        # Single column descending
        (
            "df.sort_values('column_a', ascending=False)",
            "df_pl.sort('column_a', descending=True)"
        ),
        # Multiple columns with mixed ascending
        (
            "df.sort_values(['column_a', 'column_b'], ascending=[True, False])",
            "df_pl.sort(['column_a', 'column_b'], descending=[False, True])"
        ),
        # Null handling - nulls last
        (
            "df.sort_values('column_a', na_position='last')",
            "df_pl.sort('column_a', nulls_last=True)"
        ),
        # Null handling - nulls first
        (
            "df.sort_values('column_a', na_position='first')",
            "df_pl.sort('column_a', nulls_last=False)"
        ),
        # Multiple columns with nulls last
        (
            "df.sort_values(['column_a', 'column_b'], na_position='last')",
            "df_pl.sort(['column_a', 'column_b'], nulls_last=True)"
        ),
        # Empty column list (edge case)
        (
            "df.sort_values([])",
            "df_pl.sort([])"
        ),
        # All parameters combined
        (
            "df.sort_values('column_a', ascending=False, na_position='last')",
            "df_pl.sort('column_a', descending=True, nulls_last=True)"
        ),
    ]
)
def test_sort_translations(pandas_code, expected_polars):
    """Test the translation of pandas sort_values to polars sort."""
    assert translate_test_code(pandas_code.strip()) == expected_polars.strip()

class TestComplexSortExpressions:
    """Test sorting with complex expressions.
    
    This class will be implemented later when we handle expression translations.
    Complex expressions include:
    - Column references (df['col'])
    - Arithmetic expressions
    - Function calls
    - etc.
    """
    pass
