# tests/test_fillna_translation.py
import pytest
import pandas as pd
import polars as pl
import numpy as np
from pd2pl import TranslationError
from ._helpers import compare_dataframe_ops
from tests.conftest import translate_test_code

@pytest.fixture
def df_numeric():
    return pd.DataFrame({
        'A': [1.0, np.nan, 3.0, np.nan],
        'B': [np.nan, 2.0, np.nan, 4.0]
    })

@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        'A': [1.0, np.nan, 3.0, None],
        'B': [None, 2.0, np.nan, 4.0],
        'C': ['a', None, 'c', np.nan]
    })

@pytest.fixture
def df_nulls_only():
    return pd.DataFrame({
        'A': [1.0, None, 3.0, None],
        'B': [None, 2.0, None, 4.0]
    })

@pytest.mark.parametrize(
    "pandas_code, expected_polars_method_call",
    [
        # Basic scalar value fill
        ("df.fillna(0)", ".fill_null(0).fill_nan(0)"),
        ("df.fillna(0.0)", ".fill_null(0.0).fill_nan(0.0)"),
        ("df.fillna('missing')", ".fill_null('missing').fill_nan('missing')"),
        
        # With explicit method=None
        ("df.fillna(value=0, method=None)", ".fill_null(0).fill_nan(0)"),
        
        # Dictionary-based fill
        ("df.fillna({'A': 0, 'B': 1})", ".with_columns([pl.col('A').fill_null(0).fill_nan(0), pl.col('B').fill_null(1).fill_nan(1)])"),
        
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
def test_fillna_translations(pandas_code, expected_polars_method_call, df_numeric):
    """Test translation and functional equivalence of df.fillna to fill_null."""
    # Replace 'df' with the actual dataframe name
    pandas_code_with_df = pandas_code.replace("df", "df_numeric")
    expected_full_code = f"df_numeric_pl{expected_polars_method_call}"
    
    # 1. Check translated code syntax
    translated_code = translate_test_code(pandas_code_with_df)
    assert translated_code.replace(" ", "") == expected_full_code.replace(" ", "")
    
    # 2. Check functional equivalence
    input_dfs = {'df_numeric': df_numeric}
    compare_dataframe_ops(pandas_code_with_df, translated_code, input_dfs)

@pytest.mark.parametrize(
    "pandas_code, expected_polars_method_call",
    [
        # Mixed dataframe tests
        ("df_mixed.fillna(0)", "df_mixed_pl.fill_null(0)"),
        ("df_mixed.fillna({'A': 0, 'C': 'missing'})", 
         "df_mixed_pl.with_columns([pl.col('A').fill_null(0), pl.col('C').fill_null('missing')])"),
    ]
)
def test_fillna_mixed_data(pandas_code, expected_polars_method_call, df_mixed):
    """Test fillna with mixed data types (NaN and None values)."""
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "") == expected_polars_method_call.replace(" ", "")
    
    input_dfs = {'df_mixed': df_mixed}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)

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
    assert translated_code.replace(" ", "") == expected_code.replace(" ", "")

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
