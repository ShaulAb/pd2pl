# tests/test_fillna_translation.py
import pytest
import pandas as pd
import polars as pl
import numpy as np
from pd2pl import TranslationError
from ._helpers import compare_dataframe_ops
from tests.conftest import translate_test_code
from loguru import logger

@pytest.fixture
def df_numeric():
    return pd.DataFrame({
        'A': [1.0, np.nan, 3.0, np.nan],
        'B': [np.nan, 2.0, np.nan, 4.0]
    })

@pytest.fixture
def df_strings():
    return pd.DataFrame({
        'C': ['a', None, 'c', None],
        'D': [None, 'b', None, 'd']
    })

@pytest.fixture
def df_mixed_types():
    # For type-aware tests
    return pd.DataFrame({
        'num': [1.0, np.nan, 3.0, None],
        'str': ['a', None, 'c', None]
    })

@pytest.fixture
def df_nulls_only():
    return pd.DataFrame({
        'A': [1.0, None, 3.0, None],
        'B': [None, 2.0, None, 4.0]
    })

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic scalar value fill - numeric values for numeric columns
        ("df.fillna(0)", ".fill_null(0)"),
        ("df.fillna(0.0)", ".fill_null(0.0)"),
        
        # Dictionary-based fill
        ("df.fillna({'A': 0, 'B': 1})", ".with_columns([pl.col('A').fill_null(0), pl.col('B').fill_null(1)])"),
        
        # Method-based fill
        ("df.fillna(method='ffill')", ".fill_null(strategy='forward')"),
        ("df.fillna(method='pad')", ".fill_null(strategy='forward')"),
        ("df.fillna(method='bfill')", ".fill_null(strategy='backward')"),
        ("df.fillna(method='backfill')", ".fill_null(strategy='backward')"),
        
        # Method with limit
        ("df.fillna(method='ffill', limit=2)", ".fill_null(strategy='forward', limit=2)"),
        ("df.fillna(method='bfill', limit=3)", ".fill_null(strategy='backward', limit=3)"),
        
        # Axis parameter (0 or 'index' is default)
        ("df.fillna(0, axis=0)", ".fill_null(0)"),
        ("df.fillna(0, axis='index')", ".fill_null(0)"),
        
        # Inplace parameter (converted to assignment)
        ("df.fillna(0, inplace=False)", ".fill_null(0)"),
    ]
)
def test_fillna_translations(pandas_code, expected_polars, df_numeric):
    """Test translation and functional equivalence of df.fillna to fill_null."""
    # Replace 'df' with the actual dataframe name
    pandas_code_with_df = pandas_code.replace("df", "df_numeric")
    expected_full_code = f"df_numeric_pl{expected_polars}"
    
    # 1. Check translated code syntax
    translated_code = translate_test_code(pandas_code_with_df)
    assert translated_code.replace(" ", "") == expected_full_code.replace(" ", "")
    
    # 2. Check functional equivalence
    input_dfs = {'df_numeric': df_numeric}
    compare_dataframe_ops(pandas_code_with_df, translated_code, input_dfs)

@pytest.mark.parametrize(
    "pandas_code, expected_polars_method_call",
    [
        # String fills only for string dataframes
        ("df_strings.fillna('missing')", "df_strings_pl.fill_null('missing')"),
        ("df_strings.fillna({'C': 'missing', 'D': 'unknown'})", 
         "df_strings_pl.with_columns([pl.col('C').fill_null('missing'), pl.col('D').fill_null('unknown')])"),
    ]
)
def test_fillna_string_data(pandas_code, expected_polars_method_call, df_strings):
    """Test fillna with string data types."""
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "") == expected_polars_method_call.replace(" ", "")
    
    input_dfs = {'df_strings': df_strings}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)

def test_fillna_type_aware(df_mixed_types):
    """Test fillna with type-appropriate values for mixed type dataframes."""
    # Type-compatible dictionary fill
    pandas_code = "df_mixed_types.fillna({'num': 0, 'str': 'missing'})"
    expected_code = "df_mixed_types_pl.with_columns([pl.col('num').fill_null(0), pl.col('str').fill_null('missing')])"
    
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "") == expected_code.replace(" ", "")
    
    input_dfs = {'df_mixed_types': df_mixed_types}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)

def test_fillna_type_safety_documentation():
    """Document type safety constraints in fillna translation."""
    # This test is documentation that demonstrates the type safety differences 
    # between pandas and polars
    
    # The following pandas code works but would fail in polars due to type constraints:
    # df_mixed = pd.DataFrame({'num': [1.0, np.nan], 'str': ['a', None]})
    # df_mixed.fillna(0)  # Fills both numeric and string columns with 0
    
    # In pd2pl translation, users should use dictionary with type-appropriate values:
    # correct usage: df_mixed.fillna({'num': 0, 'str': ''})
    
    # The test itself doesn't need to run a comparison, as we're just documenting behavior
    
    # Note: When pandas uses a scalar fill value for mixed types, 
    # the translation will generate code that might fail at runtime in polars
    # due to type incompatibility. Users should use dictionary-based fills with
    # type-appropriate values for mixed-type dataframes.
    assert True

@pytest.mark.parametrize(
    "pandas_code, expected_polars_method_call",
    [
        # Nulls only dataframe (should only need fill_null)
        ("df_nulls_only.fillna(0)", "df_nulls_only_pl.fill_null(0)"),
    ]
)
def test_fillna_nulls_only(pandas_code, expected_polars_method_call, df_nulls_only):
    """Test fillna with dataframe that has only None values (no NaN)."""
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "") == expected_polars_method_call.replace(" ", "")
    
    input_dfs = {'df_nulls_only': df_nulls_only}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)

def test_fillna_inplace_true():
    """Test that inplace=True is correctly translated."""
    pandas_code = "df_numeric.fillna(0, inplace=True)"
    expected_code = "df_numeric_pl = df_numeric_pl.fill_null(0)"
    
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "").strip() == expected_code.replace(" ", "")

def test_fillna_unsupported_axis():
    """Test that unsupported axis values raise TranslationError."""
    pandas_code = "df_numeric.fillna(0, axis=1)"  # axis=1 or 'columns' not supported
    
    with pytest.raises(TranslationError, match="axis=1 or 'columns' is not supported"):
        translate_test_code(pandas_code)

def test_fillna_unsupported_downcast():
    """Test that downcast parameter raises TranslationError."""
    pandas_code = "df_numeric.fillna(0, downcast='infer')"
    
    with pytest.raises(TranslationError, match="downcast parameter is not supported"):
        translate_test_code(pandas_code)
