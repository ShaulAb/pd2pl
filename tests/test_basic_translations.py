"""Test basic pandas to polars translations."""
import pytest
import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np

from tests._helpers import compare_frames
from pandas_to_polars_translator import translate_code
from tests.conftest import strip_import_lines

def test_import_stripping():
    """Test that import lines are correctly stripped."""
    test_cases = [
        # Basic cases
        (
            "import polars as pl\ndf_pl.head()",
            "df_pl.head()"
        ),
        (
            "import polars as pl\nimport polars.selectors as cs\ndf_pl.select(cs.numeric()).mean()",
            "df_pl.select(cs.numeric()).mean()"
        ),
        # Extra whitespace cases
        (
            "  import polars as pl  \ndf_pl.head()",
            "df_pl.head()"
        ),
        (
            "import polars as pl\n  import polars.selectors as cs  \ndf_pl.select(cs.numeric()).mean()",
            "df_pl.select(cs.numeric()).mean()"
        ),
        # Empty lines between imports
        (
            "import polars as pl\n\ndf_pl.head()",
            "df_pl.head()"
        ),
        (
            "import polars as pl\n\nimport polars.selectors as cs\n\ndf_pl.select(cs.numeric()).mean()",
            "df_pl.select(cs.numeric()).mean()"
        ),
        # No imports
        (
            "df_pl.head()",
            "df_pl.head()"
        ),
        # Multiple imports with mixed spacing
        (
            "import polars as pl\n  \n  import polars.selectors as cs\n\ndf_pl.select(cs.numeric()).mean()",
            "df_pl.select(cs.numeric()).mean()"
        ),
    ]
    for input_code, expected in test_cases:
        result = strip_import_lines(input_code)
        assert result == expected, f"\nInput:\n{input_code}\nExpected:\n{expected}\nGot:\n{result}"

@pytest.fixture
def df_mixed_types():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': [True, False, True, False, True]
    })

@pytest.fixture
def df_numeric_only():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1.101, 2.202, 3.303, 4.404, 5.505],
    })

class TestBasicOperations:
    """Test basic DataFrame operations."""

    def test_head(self, df_mixed_types, assert_translation):
        pandas_code = "df.head(3)"
        expected_polars = "df_pl.head(3)" # head() is the same in both styles
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_mixed_types)

    def test_tail(self, df_mixed_types, assert_translation):
        pandas_code = "df.tail(2)"
        expected_polars = "df_pl.tail(2)"  # tail() is the same in both styles
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_mixed_types)

    def test_describe(self, df_mixed_types, assert_translation):
        pandas_code = "df.describe()"
        expected_polars = "df_pl.describe()"  # describe() is the same in both styles
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_mixed_types)

class TestColumnOperations:
    """Test column selection and operations."""

    def test_single_column_select(self, df_mixed_types, assert_translation):
        pandas_code = 'df["A"]'
        expected_polars = 'df_pl["A"]'
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_mixed_types)
    
    def test_multiple_column_select(self, df_mixed_types, assert_translation):
        pandas_code = 'df[["A", "B"]]'
        expected_polars = 'df_pl[["A", "B"]]'
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_mixed_types)

    def test_column_mean(self, df_numeric_only, assert_translation):
        pandas_code = 'df["b"].mean()'
        expected_polars = 'df_pl["b"].mean()'
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_numeric_only)

class TestAggregations:
    """Test aggregation operations."""

    def test_simple_mean(self, df_numeric_only, assert_translation):
        pandas_code = "df.mean()"
        expected_polars = "df_pl.mean()"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_numeric_only)
    
    def test_simple_sum(self, df_numeric_only, assert_translation):
        pandas_code = "df.sum()"
        expected_polars = "df_pl.sum()"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_numeric_only)

    def test_numeric_only_aggregation(self, df_mixed_types, assert_translation):
        pandas_code = "df.mean(numeric_only=True)"
        expected_polars = "df_pl.select(cs.numeric()).mean()"
        translated = translate_code(pandas_code)
        assert_translation(translated, expected_polars)
        assert compare_frames(pandas_code, translated, df_mixed_types)

    def test_numeric_only_aggregations(self, df_mixed_types, assert_translation):
        """Test various aggregation methods with numeric_only=True."""
        test_cases = [
            ("df.mean(numeric_only=True)", "df_pl.select(cs.numeric()).mean()"),
            ("df.sum(numeric_only=True)", "df_pl.select(cs.numeric()).sum()"),
        ]
        for pandas_code, expected_polars in test_cases:
            translated = translate_code(pandas_code)
            assert_translation(translated, expected_polars)
            assert compare_frames(pandas_code, translated, df_mixed_types)

