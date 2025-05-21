"""Test pandas to polars translations for DataFrame.replace() method."""
import pytest
import pandas as pd
import polars as pl
import numpy as np

from tests._helpers import compare_frames
from pd2pl import translate_code
from tests.conftest import translate_test_code

@pytest.fixture
def df_for_replace():
    """Create a sample DataFrame for testing replace operations."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': [True, False, True, False, True]
    })

@pytest.fixture
def text_df_for_replace():
    """Create a sample DataFrame with text data for testing regex replacements."""
    return pd.DataFrame({
        'text': ['foo', 'bar', 'baz', 'foo bar', 'bar baz'],
        'ids': ['id1', 'id2', 'id3', 'id4', 'id5']
    })

class TestReplaceTranslations:
    """Test translations for the replace() method."""

    def test_scalar_replace(self, df_for_replace, assert_translation):
        """Test replacement of a scalar value."""
        pandas_code = "df.replace(1, 999)"
        expected_polars = "df_pl.with_columns(pl.col('*').replace(1, 999))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_for_replace)

    def test_dict_replace(self, df_for_replace, assert_translation):
        """Test replacement using a dictionary."""
        df_for_replace = df_for_replace.drop(columns=['D'])
        pandas_code = "df.replace({1: 100, 2: 200})"
        expected_polars = "df_pl.with_columns(pl.col('*').replace({1: 100, 2: 200}))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_for_replace)

    def test_list_replace(self, df_for_replace, assert_translation):
        """Test replacement using a list of values."""
        pandas_code = "df.replace([1, 2], 999)"
        expected_polars = "df_pl.with_columns(pl.col('*').replace([1, 2], 999))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_for_replace)

    def test_regex_replace(self, text_df_for_replace, assert_translation):
        """Test replacement using regular expressions."""
        pandas_code = "df.replace(to_replace=r'^ba.$', value='xyz', regex=True)"
        expected_polars = "df_pl.with_columns(pl.col('*').str.replace_all('^ba.$', 'xyz'))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, text_df_for_replace)

    def test_column_specific_replace(self, df_for_replace, assert_translation):
        """Test replacement for specific columns using a dictionary."""
        pandas_code = "df.replace({'A': {1: 100, 2: 200}})"
        expected_polars = "df_pl.with_columns(pl.col('A').replace({1: 100, 2: 200}))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_for_replace)

    # def test_none_replacement(self, df_for_replace, assert_translation):
    #     """Test replacement with None value."""
    #     # Create a DataFrame with None values
    #     df = df_for_replace.copy()
    #     df.loc[0, 'A'] = None
        
    #     pandas_code = "df.replace(pd.NA, 0)"
    #     expected_polars = "df_pl.with_columns(pl.col('*').fill_null(0))"
    #     translated = translate_code(pandas_code)
    #     assert_translation(translated, expected_polars)
    #     assert compare_frames(pandas_code, translated, df)

    def test_limit_parameter_ignored(self, df_for_replace, assert_translation):
        """Test that the deprecated limit parameter is ignored."""
        pandas_code = "df.replace(1, 999, limit=2)"
        expected_polars = "df_pl.with_columns(pl.col('*').replace(1, 999))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_for_replace) 
        # We're not testing behavior here because limit is deprecated and should be ignored

    def test_named_parameters(self, df_for_replace, assert_translation):
        """Test with explicitly named parameters."""
        pandas_code = "df.replace(to_replace=1, value=999)"
        expected_polars = "df_pl.with_columns(pl.col('*').replace(1, 999))"
        translated = translate_test_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_for_replace) 