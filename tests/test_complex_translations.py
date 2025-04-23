"""Test complex pandas to polars translations."""
import pytest
import pandas as pd
import polars as pl
import numpy as np

from tests._helpers import compare_frames
from pd2pl import translate_code

@pytest.fixture
def complex_df():
    """Create a sample DataFrame with more complex data for testing."""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B'],
        'value': [1, 2, 3, 4, 5],
        'count': [10, 20, 30, 40, 50],
        'date': pd.date_range('2023-01-01', periods=5)
    })

@pytest.fixture
def df_text():
    """Create a sample DataFrame with more complex data for testing."""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B'],
        'text': ['foo', 'bar', 'baz', 'qux', 'quux']
    })

class TestGroupByOperations:
    """Test groupby operations."""

    def test_simple_groupby(self, complex_df, assert_translation):
        pandas_code = "df.groupby('category').mean()"
        expected_polars = "df_pl.group_by('category').agg(pl.all().mean())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, complex_df)

    def test_groupby_multiple_aggs(self, complex_df, assert_translation):
        pandas_code = "df.groupby('category').agg({'value': 'mean', 'count': 'sum'})"
        expected_polars = "df_pl.group_by('category').agg([pl.col('value').mean(), pl.col('count').sum()])"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, complex_df)

class TestWindowOperations:
    """Test window operations."""

    def test_rolling_mean(self, complex_df, assert_translation):
        pandas_code = "df['value'].rolling(window=3).mean()"
        expected_polars = "df_pl.select(pl.col('value').rolling_mean(window_size=3))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, complex_df)

    def test_expanding_sum(self, complex_df, assert_translation):
        pandas_code = "df['value'].expanding().sum()"
        expected_polars = "df_pl.select(pl.col('value').cum_sum())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, complex_df)

class TestStringOperations:
    """Test string operations."""

    def test_contains(self, df_text, assert_translation):
        pandas_code = "df['category'].str.contains('A')"
        expected_polars = "df_pl.select(pl.col('category').str.contains('A'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)

    def test_string_length(self, df_text, assert_translation):
        pandas_code = "df['text'].str.len()"
        expected_polars = "df_pl.select(pl.col('text').str.len_chars())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)

class TestDateTimeOperations:
    """Test datetime operations."""

    def test_datetime_year(self, complex_df, assert_translation):
        pandas_code = "df['date'].dt.year"
        expected_polars = "df_pl.select(pl.col('date').dt.year())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, complex_df)

    def test_datetime_month(self, complex_df, assert_translation):
        pandas_code = "df['date'].dt.month"
        expected_polars = "df_pl.select(pl.col('date').dt.month())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, complex_df) 
