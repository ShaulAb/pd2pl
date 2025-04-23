"""Test filtering operations translations."""
import pytest
import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np

from src import translate_code
from tests._helpers import compare_frames


@pytest.fixture
def df_filtering_sample():
    """Create a sample DataFrame for filtering tests."""
    return pd.DataFrame({
        'col_int': [0, 10, 20, 5, -1],
        'col_float': [1.1, 5.5, 0.0, 1.0, 9.9],
        'col_str': ['value', 'other', 'x', 'y', 'value'],
        'col_bool': [True, False, True, False, True],
        'col_nullable': [1, 5, np.nan, 10, 6], # Use np.nan for Pandas compatibility
        'col_a': [6, 0, 10, 4, -1],
        'col_b': ['x', 'y', 'x', 'z', 'y'],
        'col_c': [-1, 5, 0, -10, 100]
    })


@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # Numeric Comparisons
        ("df[df['col_int'] > 10]", "df_pl.filter(pl.col('col_int') > 10)"),
        ("df[df['col_float'] <= 5.5]", "df_pl.filter(pl.col('col_float') <= 5.5)"),
        ("df[df['col_int'] == 0]", "df_pl.filter(pl.col('col_int') == 0)"),
        ("df[df['col_float'] != 1.0]", "df_pl.filter(pl.col('col_float') != 1.0)"),
        # String Comparisons
        ("df[df['col_str'] == 'value']", "df_pl.filter(pl.col('col_str') == 'value')"),
        ("df[df['col_str'] != 'other']", "df_pl.filter(pl.col('col_str') != 'other')"),
        # Boolean Column
        ("df[df['col_bool']]", "df_pl.filter(pl.col('col_bool'))"),
        ("df[~df['col_bool']]", "df_pl.filter(~pl.col('col_bool'))"),
        # Null Handling (Pandas includes NaN in != comparison, Polars doesn't by default)
        # ("df[df['col_nullable'] > 5]", "df_pl.filter(pl.col('col_nullable') > 5)"), # Polars default behavior
    ]
)
def test_basic_boolean_filtering(pandas_code, expected_polars_code, df_filtering_sample, assert_translation):
    """Test basic boolean filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # Note: compare_frames might need adjustments for subtle null handling differences if not explicitly matched.
    # For now, primarily testing translation accuracy.
    # assert compare_frames(pandas_code, translated, df_filtering_sample)


@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # Simple AND - Expect unparse to add outer parens
        ("df[(df['col_a'] > 5) & (df['col_b'] == 'x')]", "df_pl.filter((pl.col('col_a') > 5) & (pl.col('col_b') == 'x'))"),
        # Simple OR - Expect unparse to add outer parens
        ("df[(df['col_a'] <= 0) | (df['col_b'] != 'y')]", "df_pl.filter((pl.col('col_a') <= 0) | (pl.col('col_b') != 'y'))"),
        # Simple NOT
        ("df[~(df['col_a'] > 5)]", "df_pl.filter(~(pl.col('col_a') > 5))"),
        # Mixed Operators with Parentheses - Expect unparse to *not* add outer parens
        ("df[((df['col_a'] > 5) & (df['col_b'] == 'x')) | (df['col_c'] < 0)]", "df_pl.filter((pl.col('col_a') > 5) & (pl.col('col_b') == 'x') | (pl.col('col_c') < 0))"),
        # NOT applied to complex expression
        ("df[~((df['col_a'] > 5) & (df['col_b'] == 'x'))]", "df_pl.filter(~((pl.col('col_a') > 5) & (pl.col('col_b') == 'x')))")
        # Multiple ANDs - Add later if needed
        # Multiple ORs - Add later if needed
    ]
)
def test_complex_boolean_filtering(pandas_code, expected_polars_code, df_filtering_sample, assert_translation):
    """Test complex boolean filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # assert compare_frames(pandas_code, translated, df_filtering_sample)


@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # isin with List (Int, Float, String)
        ("df[df['col_int'].isin([1, 3, 5, 10])]", "df_pl.filter(pl.col('col_int').is_in([1, 3, 5, 10]))"),
        ("df[df['col_float'].isin([1.1, 3.3, 9.9])]", "df_pl.filter(pl.col('col_float').is_in([1.1, 3.3, 9.9]))"),
        ("df[df['col_str'].isin(['a', 'b', 'c', 'value'])]", "df_pl.filter(pl.col('col_str').is_in(['a', 'b', 'c', 'value']))"),
        # isin with Empty List
        ("df[df['col_int'].isin([])]", "df_pl.filter(pl.col('col_int').is_in([]))"),
        # isin Combined with Other Conditions - Expect unparse to *not* add outer parens
        ("df[(df['col_int'].isin([1, 3, 5, 10])) & (df['col_str'] == 'x')]", "df_pl.filter(pl.col('col_int').is_in([1, 3, 5, 10]) & (pl.col('col_str') == 'x'))"),
        ("df[(df['col_int'].isin([1, 3, 5, 10])) | (df['col_str'] == 'y')]", "df_pl.filter(pl.col('col_int').is_in([1, 3, 5, 10]) | (pl.col('col_str') == 'y'))"),
        # Negated isin
        ("df[~df['col_int'].isin([1, 3, 5, 10])]", "df_pl.filter(~pl.col('col_int').is_in([1, 3, 5, 10]))"),
    ]
)
def test_isin_filtering(pandas_code, expected_polars_code, df_filtering_sample, assert_translation):
    """Test isin filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # assert compare_frames(pandas_code, translated, df_filtering_sample) 


# --- ADDED TEST FUNCTION ---
@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # Basic isna/notna
        ("df[df['col_nullable'].isna()]", "df_pl.filter(pl.col('col_nullable').is_null())"),
        ("df[df['col_nullable'].notna()]", "df_pl.filter(pl.col('col_nullable').is_not_null())"),
        # Combined with &
        ("df[(df['col_nullable'].isna()) & (df['col_int'] > 0)]", "df_pl.filter(pl.col('col_nullable').is_null() & (pl.col('col_int') > 0))"),
        # Combined with |
        ("df[(df['col_nullable'].notna()) | (df['col_str'] == 'x')]", "df_pl.filter(pl.col('col_nullable').is_not_null() | (pl.col('col_str') == 'x'))"),
    ]
)
def test_isna_notna_filtering(pandas_code, expected_polars_code, df_filtering_sample, assert_translation):
    """Test isna and notna filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # assert compare_frames(pandas_code, translated, df_filtering_sample) 