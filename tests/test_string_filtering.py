"""Test string filtering operations within filter()."""
import pytest
import pandas as pd
import polars as pl
import numpy as np

from pd2pl import translate_code
from tests._helpers import compare_frames

@pytest.fixture
def df_string_filtering_sample():
    """Create a sample DataFrame for string filtering tests."""
    return pd.DataFrame({
        'text_col': ['substring', 'no match', 'another substring here', None, 'subxxxing'],
        'prefix_col': ['pre_value', 'pre_other', 'no_pre', None, 'pre_again'],
        'suffix_col': ['value_fix', 'other_fix', 'no_fix_at_end', None, 'ends_with_fix']
    })

@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # Basic contains
        ("df[df['text_col'].str.contains('substring')]", "df_pl.filter(pl.col('text_col').str.contains('substring'))"),
        # Contains combined with &
        ("df[(df['text_col'].str.contains('sub')) & (df['prefix_col'] == 'pre_value')]", "df_pl.filter(pl.col('text_col').str.contains('sub') & (pl.col('prefix_col') == 'pre_value'))"),
        # Contains with regex=True (default in Polars)
        ("df[df['text_col'].str.contains('sub.*ing')]", "df_pl.filter(pl.col('text_col').str.contains('sub.*ing'))"),
        # Contains with na=False (needs translation)
        # ("df[df['text_col'].str.contains('sub', na=False)]", "df_pl.filter(pl.col('text_col').str.contains('sub').fill_null(False))"),
    ]
)
def test_string_contains_filtering(pandas_code, expected_polars_code, df_string_filtering_sample, assert_translation):
    """Test string contains filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # assert compare_frames(pandas_code, translated, df_string_filtering_sample)

@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # Basic startswith
        ("df[df['prefix_col'].str.startswith('pre')]", "df_pl.filter(pl.col('prefix_col').str.starts_with('pre'))"),
        # Startswith combined with |
        ("df[(df['prefix_col'].str.startswith('pre')) | (df['suffix_col'] == 'other_fix')]", "df_pl.filter(pl.col('prefix_col').str.starts_with('pre') | (pl.col('suffix_col') == 'other_fix'))"),
        # Startswith with na=False (needs translation)
        # ("df[df['prefix_col'].str.startswith('pre', na=False)]", "df_pl.filter(pl.col('prefix_col').str.starts_with('pre').fill_null(False))"),
    ]
)
def test_string_startswith_filtering(pandas_code, expected_polars_code, df_string_filtering_sample, assert_translation):
    """Test string startswith filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # assert compare_frames(pandas_code, translated, df_string_filtering_sample)

@pytest.mark.parametrize(
    "pandas_code, expected_polars_code",
    [
        # Basic endswith
        ("df[df['suffix_col'].str.endswith('fix')]", "df_pl.filter(pl.col('suffix_col').str.ends_with('fix'))"),
        # Endswith combined with &
        ("df[(df['suffix_col'].str.endswith('fix')) & (df['prefix_col'] == 'pre_other')]", "df_pl.filter(pl.col('suffix_col').str.ends_with('fix') & (pl.col('prefix_col') == 'pre_other'))"),
        # Endswith with na=False (needs translation)
        # ("df[df['suffix_col'].str.endswith('fix', na=False)]", "df_pl.filter(pl.col('suffix_col').str.ends_with('fix').fill_null(False))"),
    ]
)
def test_string_endswith_filtering(pandas_code, expected_polars_code, df_string_filtering_sample, assert_translation):
    """Test string endswith filtering translations."""
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars_code)
    # assert compare_frames(pandas_code, translated, df_string_filtering_sample) 