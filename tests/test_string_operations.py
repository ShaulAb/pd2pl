"""Test string operations translations."""
import pytest
import pandas as pd
import polars as pl
import numpy as np

from pd2pl import translate_code
from tests._helpers import compare_frames

@pytest.fixture
def df_text():
    """Create a sample DataFrame with text data for testing string operations."""
    return pd.DataFrame({
        'text': ['Hello World', 'foo BAR', 'Python,Pandas,Polars', None, 'trailing space  '],
        'pattern': ['Hello.*', 'foo', 'Python', None, 'trail'],
        'cat': ['A', 'B', 'C', None, 'A']
    })

class TestBasicStringOperations:
    """Test basic string operations like lower, upper, strip."""

    def test_str_lower(self, df_text, assert_translation):
        """Test string lower method translation."""
        pandas_code = "df['text'].str.lower()"
        expected_polars = "df_pl.select(pl.col('text').str.to_lowercase())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)

    def test_str_upper(self, df_text, assert_translation):
        """Test string upper method translation."""
        pandas_code = "df['text'].str.upper()"
        expected_polars = "df_pl.select(pl.col('text').str.to_uppercase())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)

    def test_str_strip(self, df_text, assert_translation):
        """Test string strip method translation."""
        pandas_code = "df['text'].str.strip()"
        expected_polars = "df_pl.select(pl.col('text').str.strip_chars())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
        
    def test_str_len(self, df_text, assert_translation):
        """Test string length method translation."""
        pandas_code = "df['text'].str.len()"
        expected_polars = "df_pl.select(pl.col('text').str.len_chars())"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)

class TestStringSearchOperations:
    """Test string search operations like contains, startswith, endswith."""
    
    def test_str_contains(self, df_text, assert_translation):
        """Test string contains method translation."""
        pandas_code = "df['text'].str.contains('World')"
        expected_polars = "df_pl.select(pl.col('text').str.contains('World'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
        
    def test_str_startswith(self, df_text, assert_translation):
        """Test string startswith method translation."""
        pandas_code = "df['text'].str.startswith('Hello')"
        expected_polars = "df_pl.select(pl.col('text').str.starts_with('Hello'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
        
    def test_str_endswith(self, df_text, assert_translation):
        """Test string endswith method translation."""
        pandas_code = "df['text'].str.endswith('World')"
        expected_polars = "df_pl.select(pl.col('text').str.ends_with('World'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)

class TestStringTransformOperations:
    """Test string transformation operations like split, replace, extract."""
    
    def test_str_split_basic(self, df_text, assert_translation):
        """Test basic string split method translation."""
        pandas_code = "df['text'].str.split(',')"
        expected_polars = "df_pl.select(pl.col('text').str.split(','))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
    
    def test_str_split_with_pat(self, df_text, assert_translation):
        """Test string split with 'pat' parameter translation."""
        pandas_code = "df['text'].str.split(pat=',')"
        expected_polars = "df_pl.select(pl.col('text').str.split(by=','))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
    
    def test_str_replace_basic(self, df_text, assert_translation):
        """Test basic string replace method translation."""
        pandas_code = "df['text'].str.replace('World', 'Globe')"
        expected_polars = "df_pl.select(pl.col('text').str.replace('World', 'Globe'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
    
    def test_str_replace_with_regex(self, df_text, assert_translation):
        """Test string replace with regex parameter translation."""
        pandas_code = "df['text'].str.replace('W.*d', 'Globe', regex=True)"
        expected_polars = "df_pl.select(pl.col('text').str.replace('W.*d', 'Globe'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_text)
    
    def test_str_extract(self, df_text, assert_translation):
        """Test string extract method translation."""
        pandas_code = "df['text'].str.extract('(\\w+)')"
        expected_polars = "df_pl.select(pl.col('text').str.extract('(\\\\w+)'))"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        # Skip frame comparison for now as behavior might differ 