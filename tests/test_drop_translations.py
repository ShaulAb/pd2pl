"""Test drop operation translations."""
import pytest
from pd2pl import translate_code

@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic single column drop
        (
            "df.drop('foo')",
            "df_pl.drop('foo')"
        ),
        # Drop multiple columns using list
        (
            "df.drop(['foo', 'bar'])",
            "df_pl.drop(['foo', 'bar'])"
        ),
        # Drop with strict=False
        (
            "df.drop('foo', strict=False)",
            "df_pl.drop('foo', strict=False)"
        ),
        # Drop multiple columns using positional args
        (
            "df.drop('foo', 'bar')",
            "df_pl.drop('foo', 'bar')"
        ),
        # Drop with axis=1 (columns) - should be simplified
        (
            "df.drop('foo', axis=1)",
            "df_pl.drop('foo')"
        ),
        # Drop with axis='columns' - should be simplified
        (
            "df.drop('foo', axis='columns')",
            "df_pl.drop('foo')"
        ),
        # Drop with columns parameter - should be simplified
        (
            "df.drop(columns='foo')",
            "df_pl.drop('foo')"
        ),
        # Drop with columns parameter as list - should be simplified
        (
            "df.drop(columns=['foo', 'bar'])",
            "df_pl.drop(['foo', 'bar'])"
        ),
    ]
)
def test_drop_translations(pandas_code, expected_polars):
    """Test the translation of pandas drop to polars drop."""
    assert translate_code(pandas_code.strip()) == expected_polars.strip()

class TestComplexDropExpressions:
    """Test dropping with complex expressions.
    
    This class will be implemented later when we handle expression translations.
    Complex expressions include:
    - Column selectors (df.drop(pl.selectors.numeric()))
    - Column references (df.drop(df.columns[1:]))
    - etc.
    """
    pass 