"""Test rename operation translations."""
import pytest
from pd2pl import translate_code

@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic dictionary rename
        (
            "df.rename({'foo': 'apple'})",
            "df_pl.rename({'foo': 'apple'})"
        ),
        # Multiple columns rename
        (
            "df.rename({'foo': 'apple', 'bar': 'banana'})",
            "df_pl.rename({'foo': 'apple', 'bar': 'banana'})"
        ),
        # Rename with strict=False
        (
            "df.rename({'foo': 'apple'}, strict=False)",
            "df_pl.rename({'foo': 'apple'}, strict=False)"
        ),
        # Function-based rename
        (
            "df.rename(lambda x: 'c' + x[1:])",
            "df_pl.rename(lambda x: 'c' + x[1:])"
        ),
    ]
)
def test_rename_translations(pandas_code, expected_polars):
    """Test the translation of pandas rename to polars rename."""
    assert translate_code(pandas_code.strip()) == expected_polars.strip()

class TestComplexRenameExpressions:
    """Test renaming with complex expressions.
    
    This class will be implemented later when we handle expression translations.
    Complex expressions include:
    - Column references (df['col'])
    - String operations
    - Function calls
    - etc.
    """
    pass 