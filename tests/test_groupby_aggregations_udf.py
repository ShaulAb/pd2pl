import pytest
import pandas as pd
import polars as pl

from tests._helpers import compare_frames
from tests.conftest import translate_test_code
@pytest.fixture
def df_groupby_udf():
    return pd.DataFrame({
        'cat': ['A', 'A', 'B', 'B', 'C', 'C'],
        'val': [1, 2, 3, 4, 5, 6],
    })

def test_groupby_agg_simple_lambda(df_groupby_udf):
    pandas_code = "df.groupby('cat').agg({'val': lambda x: x.max() - x.min()})"
    expected_polars = (
        "df_pl.group_by('cat').agg([((pl.col('val').max().alias('val_max') - pl.col('val').min().alias('val_min')).alias('val_udf')])"
    )
    translated = translate_test_code(pandas_code)
    assert translated == expected_polars
    # Frame comparison is not possible unless the translation engine can execute the lambda


def test_groupby_agg_named_function(df_groupby_udf):
    pandas_code = "df.groupby('cat').agg({'val': np.mean})"
    expected_polars = "df_pl.group_by('cat').agg([pl.col('val').mean().alias('val_mean')])"
    translated = translate_test_code(pandas_code)
    assert translated == expected_polars


def test_groupby_agg_unsupported_lambda(df_groupby_udf):
    pandas_code = "df.groupby('cat').agg({'val': lambda x: x.sort_values()})"
    with pytest.raises(Exception) as excinfo:
        translate_test_code(pandas_code)
    assert 'Only simple single-expression lambdas using supported methods' in str(excinfo.value)


def test_groupby_agg_unsupported_function(df_groupby_udf):
    pandas_code = "df.groupby('cat').agg({'val': custom_func})"
    with pytest.raises(Exception) as excinfo:
        translate_test_code(pandas_code)
    assert 'Unsupported named function' in str(excinfo.value) 