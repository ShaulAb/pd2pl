# tests/test_melt_translation.py
import pytest
import pandas as pd
import polars as pl
from pd2pl import translate_code
from tests._helpers import compare_frames

@pytest.fixture
def wide_df():
    """Fixture for melt tests."""
    return pd.DataFrame({
        'A': ['id1', 'id2', 'id3'],
        'B': [10, 20, 30],
        'C': [1.1, 2.2, 3.3],
        'D': [True, False, True]
    })

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Basic melt with id_vars only (infer value_vars)
        (
            "df.melt(id_vars=['A'])",
            "df_pl.unpivot(index=['A'])"
        ),
        # Melt with single id_var and single value_var
        (
            "df.melt(id_vars='A', value_vars='B')",
            "df_pl.unpivot(index='A', on='B')"
        ),
        # Melt with list id_vars and list value_vars
        (
            "df.melt(id_vars=['A', 'B'], value_vars=['C', 'D'])",
            "df_pl.unpivot(index=['A', 'B'], on=['C', 'D'])"
        ),
        # Custom variable name
        (
            "df.melt(id_vars='A', var_name='myVariable')",
            "df_pl.unpivot(index='A', variable_name='myVariable')"
        ),
        # Custom value name
        (
            "df.melt(id_vars='A', value_name='myValue')",
            "df_pl.unpivot(index='A', value_name='myValue')"
        ),
        # Custom variable and value names
        (
            "df.melt(id_vars=['A', 'B'], value_vars='C', var_name='var', value_name='val')",
            "df_pl.unpivot(index=['A', 'B'], on='C', variable_name='var', value_name='val')"
        ),
        # Ignore col_level (not applicable)
        (
            "df.melt(id_vars='A', col_level=0)",
            "df_pl.unpivot(index='A')"
        ),
        # Ignore ignore_index (not applicable)
        (
            "df.melt(id_vars='A', ignore_index=False)", # Default is True, test False too
            "df_pl.unpivot(index='A')"
        ),
    ]
)
def test_melt_translations(pandas_code, expected_polars, wide_df):
    """Test the translation and functional equivalence of pandas melt to polars unpivot."""
    translated_polars = translate_code(pandas_code.strip())
    assert translated_polars == expected_polars.strip()
    assert compare_frames(pandas_code.strip(), translated_polars, wide_df), \
           f"DataFrame comparison failed for:\nPandas: {pandas_code}\nPolars: {translated_polars}" 