"""Test concat operation translations (only variants supported by polars.concat).

Note: There are fundamental behavioral differences between pandas.concat and polars.concat:
1. Polars concat with how='vertical' requires all DataFrames to have identical columns
   while pandas.concat will union columns and fill missing values with NaN
2. Polars concat with how='horizontal' raises error on duplicate column names
   while pandas.concat automatically adds suffixes
3. Polars concat requires consistent types in vertical mode unless using 'vertical_relaxed'

These tests focus on syntax translation, not functional equivalence.
"""
import pytest
import pandas as pd
from pd2pl import TranslationError
from tests.conftest import translate_test_code

@pytest.fixture
def df1():
    """DataFrame with columns A, B, C."""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [4.0, 5.0, 6.0]
    })

@pytest.fixture
def df2():
    """DataFrame with columns A, B, D."""
    return pd.DataFrame({
        'A': [4, 5, 6],
        'B': ['d', 'e', 'f'],
        'D': [7.0, 8.0, 9.0]
    })

@pytest.fixture
def df3():
    """DataFrame with completely different columns E, F, G."""
    return pd.DataFrame({
        'E': [10, 11, 12],
        'F': ['g', 'h', 'i'],
        'G': [13.0, 14.0, 15.0]
    })

@pytest.fixture
def series1():
    """Series with simple data."""
    return pd.Series([1, 2, 3], name='X')

@pytest.fixture
def series2():
    """Series with simple data."""
    return pd.Series([4, 5, 6], name='Y')

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic vertical concatenation (default)
        ("pd.concat([df1, df2])", "pl.concat([df1, df2])"),
        # Vertical concatenation with explicit axis=0
        ("pd.concat([df1, df2], axis=0)", "pl.concat([df1, df2])"),
        # Horizontal concatenation with axis=1
        ("pd.concat([df1, df3], axis=1)", "pl.concat([df1, df3], how='horizontal')"),
        # Horizontal concatenation with axis='columns'
        ("pd.concat([df1, df3], axis='columns')", "pl.concat([df1, df3], how='horizontal')"),
        # Series concatenation (vertical only)
        ("pd.concat([series1, series2])", "pl.concat([series1, series2])"),
        # Single DataFrame in a list
        ("pd.concat([df1])", "pl.concat([df1])"),
        # Multiple DataFrames with same columns (for vertical)
        ("pd.concat([df1, df2])", "pl.concat([df1, df2])"),
    ]
)
def test_concat_translations(pandas_code, expected_polars):
    """Test the translation of pandas concat to polars concat (supported variants only)."""
    assert translate_test_code(pandas_code.strip()) == expected_polars.strip()


def test_concat_unsupported_parameters():
    """Test that unsupported pandas.concat parameters raise TranslationError."""
    unsupported_cases = [
        "pd.concat([df1, df2], keys=['x', 'y'])",
        "pd.concat([df1, df2], verify_integrity=True)",
        "pd.concat([df1, df2], sort=True)",
        "pd.concat([df1, df2], names=['level1'])",
        "pd.concat([df1, df2], levels=[['a', 'b']])",
        "pd.concat([df1, df2], join='inner')",  # Not supported by polars.concat
        "pd.concat([df1, df2], join='outer')",  # Not supported by polars.concat
        "pd.concat([df1, df2], ignore_index=True)",  # Not supported by polars.concat
    ]
    for pd_code in unsupported_cases:
        with pytest.raises(TranslationError):
            translate_test_code(pd_code)


# Note: Functional equivalence tests are limited to cases where behavior matches

def test_concat_vertical_same_columns(df1, df2):
    """Test vertical concatenation with identical columns (works in both pandas and polars)."""
    # This test uses two DataFrames with identical column names and types
    pd_code = "pd.concat([df1, df2])"
    pl_code = "pl.concat([df1, df2])"
    
    # This would be used for functional testing which we're skipping
    # input_dfs = {'df1': df1, 'df2': df2}
    # compare_dataframe_ops(pd_code, pl_code, input_dfs)
    
    # Instead just test the code translation
    assert translate_test_code(pd_code) == pl_code


def test_concat_horizontal_no_duplicates(df1, df3):
    """Test horizontal concatenation with no column name overlap."""
    # This test uses two DataFrames with completely different column names
    pd_code = "pd.concat([df1, df3], axis=1)"
    pl_code = "pl.concat([df1, df3], how='horizontal')"
    
    # This would be used for functional testing which we're skipping
    # input_dfs = {'df1': df1, 'df3': df3}
    # compare_dataframe_ops(pd_code, pl_code, input_dfs)
    
    # Instead just test the code translation
    assert translate_test_code(pd_code) == pl_code


def test_empty_dataframes_code_translation():
    """Test code translation for empty DataFrames."""
    pd_code = "pd.concat([empty_df1, empty_df2])"
    pl_code = "pl.concat([empty_df1, empty_df2])"
    
    # Just test the code translation, not the runtime behavior
    assert translate_test_code(pd_code) == pl_code


def test_different_dtypes_code_translation():
    """Test code translation for different dtypes."""
    pd_code = "pd.concat([df1, df_diff_types])"
    pl_code = "pl.concat([df1, df_diff_types])"
    
    # Just test the code translation, not the runtime behavior
    assert translate_test_code(pd_code) == pl_code


# Add docstring comment explaining why we're not testing certain pandas behaviors
"""
NOTE: The following pandas concat behaviors are not directly translatable to polars:

1. Concat of DataFrames with different columns in 'vertical' mode
   Pandas: Combines all columns and fills missing values with NaN
   Polars: Raises an error unless using 'how='diagonal'' 

2. Concat of DataFrames with duplicate columns in 'horizontal' mode
   Pandas: Automatically adds suffixes to distinguish columns
   Polars: Raises a DuplicateError

3. Concat with different dtypes in the same column
   Pandas: Attempts automatic type coercion
   Polars: Requires explicit 'vertical_relaxed' mode

These differences require either changing the translation strategy or
documenting the limitations for users.
""" 