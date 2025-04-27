import pytest
from pd2pl import translate_code

@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # astype('category') on a DataFrame column
        (
            "df['col'].astype('category')",
            "df_pl['col'].cast(pl.Categorical)"
        ),
        # Series creation with dtype='category'
        (
            "pd.Series(data, dtype='category')",
            "pl.Series(data, dtype=pl.Categorical)"
        ),
        # DataFrame creation with dtype='category' (should apply to all columns)
        (
            "pd.DataFrame(data, dtype='category')",
            "pl.DataFrame(data, dtype=pl.Categorical)"
        ),
        # Integer dtypes
        (
            "df['col'].astype('int64')",
            "df_pl['col'].cast(pl.Int64)"
        ),
        (
            "pd.Series(data, dtype='int32')",
            "pl.Series(data, dtype=pl.Int32)"
        ),
        (
            "pd.DataFrame(data, dtype='Int16')",
            "pl.DataFrame(data, dtype=pl.Int16)"
        ),
        # Float dtypes
        (
            "df['col'].astype('float32')",
            "df_pl['col'].cast(pl.Float32)"
        ),
        (
            "pd.Series(data, dtype='float64')",
            "pl.Series(data, dtype=pl.Float64)"
        ),
        (
            "pd.DataFrame(data, dtype='float')",
            "pl.DataFrame(data, dtype=pl.Float64)"
        ),
        # Boolean dtypes
        (
            "df['col'].astype('bool')",
            "df_pl['col'].cast(pl.Boolean)"
        ),
        (
            "pd.Series(data, dtype='boolean')",
            "pl.Series(data, dtype=pl.Boolean)"
        ),
        (
            "pd.DataFrame(data, dtype='bool')",
            "pl.DataFrame(data, dtype=pl.Boolean)"
        ),
        # Placeholder for future: astype('Int64')
        # (
        #     "df['col'].astype('Int64')",
        #     "df_pl['col'].cast(pl.Int64)"
        # ),
    ]
)
def test_dtype_translation(pandas_code, expected_polars):
    """Test translation of pandas dtype usage to polars dtype usage."""
    translated = translate_code(pandas_code, postprocess_imports=True)
    assert expected_polars in translated 