"""
Tests for method chain transformations in the astroid-based transformer.

These tests verify that method chains are correctly transformed from pandas to polars syntax,
with proper schema tracking and type inference throughout the chain.
"""

import pytest
import pandas as pd
import polars as pl
from tests.conftest import translate_test_code
from tests._helpers import compare_dataframe_ops

# Fixtures
@pytest.fixture
def df_basic():
    """Basic test DataFrame with numeric and string columns."""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['x', 'y', 'z']
    })

@pytest.fixture
def df_group():
    """Test DataFrame with grouping column and values."""
    return pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'values': [1, 2, 3, 4],
        'flag': [True, False, True, False]
    })

# Test cases for basic method chains
@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic selection and renaming
        (
            "df.filter(items=['A']).rename(columns={'A': 'C'})",
            "df_pl.select(['A']).rename({'A': 'C'})"
        ),
        # Filter and select
        (
            "df[df.A > 1].filter(items=['B'])",
            "df_pl.filter(pl.col('A') > 1).select(['B'])"
        ),
        # Multiple operations
        (
            "df.rename(columns={'A': 'X'}).sort_values('B')",
            "df_pl.rename({'A': 'X'}).sort('B')"
        ),
    ]
)
def test_basic_method_chains(pandas_code, expected_polars, df_basic):
    """Test translation of basic method chains."""
    # Replace 'df' with the actual dataframe name
    pandas_code = f"result = {pandas_code}".replace("df", "df_basic")
    expected_code = f"result = {expected_polars}".replace("df_pl", "df_basic_pl")
    
    # 1. Check translated code syntax
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "") == expected_code.replace(" ", "")
    
    # 2. Check functional equivalence
    input_dfs = {'df_basic': df_basic}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)

# Test cases for groupby operations
@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic groupby-agg
        (
            "df.groupby('category').agg({'values': 'mean'}).reset_index()",
            "df_pl.group_by('category').agg(pl.col('values').mean().alias('values_mean'))"
        ),
        # Multiple aggregations
        (
            "df.groupby('category').agg({'values': ['mean', 'sum']}).reset_index()",
            "df_pl.group_by('category').agg(["
            "pl.col('values').mean().alias('values_mean'), "
            "pl.col('values').sum().alias('values_sum')"
            "])"
        ),
        # Groupby with filter
        (
            "df[df.flag].groupby('category').agg({'values': 'mean'}).reset_index()",
            "df_pl.filter(pl.col('flag'))"
            ".group_by('category').agg(pl.col('values').mean().alias('values_mean'))"
        ),
    ]
)
def test_groupby_chains(pandas_code, expected_polars, df_group):
    """Test translation of groupby-aggregation chains."""
    # Replace 'df' with the actual dataframe name
    pandas_code = f"result = {pandas_code}".replace("df.", "df_group.")
    expected_code = f"result = {expected_polars}".replace("df_pl", "df_group_pl")
    
    # 1. Check translated code syntax
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "") == expected_code.replace(" ", "")
    
    # 2. Check functional equivalence
    input_dfs = {'df_group': df_group}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)

# Test cases for complex method chains
@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Complex chain with filter, groupby, and sort
        (
            """(
            df[df.values > 1]
            .groupby('category')
            .agg({'values': ['mean', 'count']})
            .reset_index()
            .sort_values(('values', 'mean'), ascending=False)
            )""",
            """(
            df_pl.filter(pl.col('values') > 1)
            .group_by('category')
            .agg([
                pl.col('values').mean().alias('values_mean'),
                pl.col('values').count().alias('values_count')
            ])
            .sort('values_mean', descending=True)
            )"""
        ),
    ]
)
def test_complex_chains(pandas_code, expected_polars, df_group):
    """Test translation of complex method chains."""
    # Replace 'df' with the actual dataframe name
    pandas_code = f"result = {pandas_code}".replace("df[", "df_group[")
    expected_code = f"result = {expected_polars}".replace("df_pl", "df_group_pl")
    
    # 1. Check translated code syntax
    translated_code = translate_test_code(pandas_code)
    assert translated_code.replace(" ", "").replace("\n", "") == \
           expected_code.replace(" ", "").replace("\n", "")
    
    # 2. Check functional equivalence
    input_dfs = {'df_group': df_group}
    compare_dataframe_ops(pandas_code, translated_code, input_dfs)
