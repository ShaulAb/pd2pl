# tests/test_merge_translation.py
import pytest
import pandas as pd
import polars as pl
from pd2pl import TranslationError
from ._helpers import compare_dataframe_ops # IMPORT HELPER
from tests.conftest import translate_test_code

@pytest.fixture
def df_left():
    return pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'A': ['A0', 'A1', 'A2'], 'common': [1, 2, 3]})

@pytest.fixture
def df_right():
    return pd.DataFrame({'key': ['K0', 'K1', 'K3'], 'B': ['B0', 'B1', 'B3'], 'common': [4, 5, 6]})

@pytest.fixture
def df_right_diffkey():
     return pd.DataFrame({'key_right': ['K0', 'K1', 'K3'], 'B': ['B0', 'B1', 'B3'], 'common': [4, 5, 6]})

@pytest.mark.parametrize(
    "pandas_code, expected_polars_method_call", # Test structure: df_left_pl.<expected>
    [
        # Basic column join (on)
        ( "pd.merge(df_left, df_right, on='key', how='inner')", ".join(df_right_pl, on='key', how='inner', coalesce=True)"),
        ( "pd.merge(df_left, df_right, on='key', how='left')", ".join(df_right_pl, on='key', how='left', coalesce=True)"),
        ( "pd.merge(df_left, df_right, on=['key'], how='right')", ".join(df_right_pl, on=['key'], how='right', coalesce=True)"),
        ( "pd.merge(df_left, df_right, on='key', how='outer')", ".join(df_right_pl, on='key', how='full', coalesce=True)"), # outer -> full
        # Different keys
        ( "pd.merge(df_left, df_right_diffkey, left_on='key', right_on='key_right', how='inner')", ".join(df_right_diffkey_pl, left_on='key', right_on='key_right', how='inner', coalesce=True)"),
        # rsuffix only
        ( "pd.merge(df_left, df_right, on='key', suffixes=('', '_R'))", ".join(df_right_pl, on='key', how='inner', suffix='_R', coalesce=True)"),
        # default suffixes (NO suffix expected in Polars code if not explicit in pandas)
        ( "pd.merge(df_left, df_right, on='key')", ".join(df_right_pl, on='key', how='inner', coalesce=True)"), # MODIFIED: Removed suffix='_right'
        # Validation
        ( "pd.merge(df_left, df_right, on='key', validate='1:1')", ".join(df_right_pl, on='key', how='inner', validate='1:1', coalesce=True)"),
        ( "pd.merge(df_left, df_right, on='key', validate='m:1')", ".join(df_right_pl, on='key', how='inner', validate='m:1', coalesce=True)"),
    ]
)
def test_merge_column_join_translations(pandas_code, expected_polars_method_call, df_left, df_right, df_right_diffkey):
    """Test translation and functional equivalence of column-based pd.merge to df.join."""
    translated_code = translate_test_code(pandas_code)

    # 1. Check generated code syntax
    expected_full_code = f"df_left_pl{expected_polars_method_call}"
    assert translated_code.replace(" ", "") == expected_full_code.replace(" ", "")

    # 2. Check functional equivalence
    input_dfs = {
        'df_left': df_left,
        'df_right': df_right,
        'df_right_diffkey': df_right_diffkey
    } # MODIFIED: Create dict for helper
    compare_dataframe_ops(pandas_code, translated_code, input_dfs) # MODIFIED: Call helper

def test_merge_unsupported_index_join():
    """Test that index joins raise TranslationError."""
    pd_code_left = "pd.merge(df_left, df_right, left_index=True, right_on='key')"
    pd_code_right = "pd.merge(df_left, df_right, left_on='key', right_index=True)"
    pd_code_both = "pd.merge(df_left, df_right, left_index=True, right_index=True)"
    with pytest.raises(TranslationError, match="Index joins .* not yet supported"):
        translate_test_code(pd_code_left)
    with pytest.raises(TranslationError, match="Index joins .* not yet supported"):
        translate_test_code(pd_code_right)
    with pytest.raises(TranslationError, match="Index joins .* not yet supported"):
        translate_test_code(pd_code_both)

def test_merge_unsupported_lsuffix():
    """Test that lsuffix raises TranslationError."""
    pd_code = "pd.merge(df_left, df_right, on='key', suffixes=('_L', '_R'))"
    # Use slightly less strict regex to match the error message
    with pytest.raises(TranslationError, match="lsuffix .* is not supported"): # MODIFIED
        translate_test_code(pd_code) 