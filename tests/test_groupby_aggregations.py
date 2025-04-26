import pytest
import pandas as pd
import polars as pl
from pd2pl import translate_code
from tests._helpers import compare_frames

@pytest.fixture
def df_groupby_mixed():
    """Fixture for groupby aggregation tests: two categorical, two numerical columns."""
    return pd.DataFrame({
        'cat1': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B'],
        'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
        'val1': [1, 2, 3, 4, 5, 6, 7, 8],
        'val2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    })

def assert_translation(translated, expected):
    assert translated == expected, f"\nExpected:\n{expected}\nGot:\n{translated}"

@pytest.mark.parametrize("agg_func", ["sum", "mean", "min", "max", "count", "median"])
def test_single_column_groupby_builtin_aggs(df_groupby_mixed, agg_func):
    pandas_code = f"df.groupby('cat1').{agg_func}()"
    expected_polars = f"df_pl.groupby('cat1').{agg_func}()"
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars)
    assert compare_frames(pandas_code, translated, df_groupby_mixed)

@pytest.mark.parametrize("agg_func", ["sum", "mean", "min", "max", "count", "median"])
def test_multi_column_groupby_builtin_aggs(df_groupby_mixed, agg_func):
    pandas_code = f"df.groupby(['cat1', 'cat2']).{agg_func}()"
    expected_polars = f"df_pl.groupby(['cat1', 'cat2']).{agg_func}()"
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars)
    assert compare_frames(pandas_code, translated, df_groupby_mixed)

def test_named_aggregation_dict_style(df_groupby_mixed):
    pandas_code = "df.groupby('cat1').agg({'val1': 'sum', 'val2': 'mean'})"
    expected_polars = (
        "df_pl.groupby('cat1').agg([pl.col('val1').sum().alias('val1_sum'), "
        "pl.col('val2').mean().alias('val2_mean')])"
    )
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars)
    assert compare_frames(pandas_code, translated, df_groupby_mixed)

def test_named_aggregation_tuple_style(df_groupby_mixed):
    pandas_code = (
        "df.groupby('cat1').agg(val1_sum=('val1', 'sum'), val2_mean=('val2', 'mean'))"
    )
    expected_polars = (
        "df_pl.groupby('cat1').agg([pl.col('val1').sum().alias('val1_sum'), "
        "pl.col('val2').mean().alias('val2_mean')])"
    )
    translated = translate_code(pandas_code)
    assert_translation(translated, expected_polars)
    assert compare_frames(pandas_code, translated, df_groupby_mixed) 