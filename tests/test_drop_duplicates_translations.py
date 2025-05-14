import pytest
from pd2pl import translate_code
from tests.conftest import translate_test_code

@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Default (keep='first') -> keep='first', maintain_order=True
        (
            "df.drop_duplicates()",
            "df_pl.unique(maintain_order=True, keep='first')"
        ),
        # Explicit keep='first' -> keep='first', maintain_order=True
        (
            "df.drop_duplicates(keep='first')",
            "df_pl.unique(maintain_order=True, keep='first')"
        ),
        # keep='last' -> keep='last', maintain_order=True
        (
            "df.drop_duplicates(keep='last')",
            "df_pl.unique(maintain_order=True, keep='last')"
        ),
        # keep=False -> keep='none', maintain_order=True
        (
            "df.drop_duplicates(keep=False)",
            "df_pl.unique(maintain_order=True, keep='none')"
        ),
        # With subset list -> subset list, keep='first', maintain_order=True
        (
            "df.drop_duplicates(subset=['a', 'b'])",
            "df_pl.unique(subset=['a', 'b'], maintain_order=True, keep='first')"
        ),
        # With subset string -> subset string, keep='first', maintain_order=True
        (
            "df.drop_duplicates(subset='a')",
            "df_pl.unique(subset='a', maintain_order=True, keep='first')"
        ),
        # Subset and keep='last' -> subset, keep='last', maintain_order=True
        (
            "df.drop_duplicates(subset='col_a', keep='last')",
            "df_pl.unique(subset='col_a', maintain_order=True, keep='last')"
        ),
        # Subset and keep=False -> subset, keep='none', maintain_order=True
        (
            "df.drop_duplicates(subset=['col_a', 'col_b'], keep=False)",
            "df_pl.unique(subset=['col_a', 'col_b'], maintain_order=True, keep='none')"
        ),
        # Ignore inplace=True -> keep='first', maintain_order=True (assignment handled elsewhere)
        (
            "df.drop_duplicates(inplace=True)",
            "df_pl.unique(maintain_order=True, keep='first')" # Translator handles assignment
        ),
        # Ignore ignore_index=True -> keep='first', maintain_order=True
        (
            "df.drop_duplicates(ignore_index=True)",
            "df_pl.unique(maintain_order=True, keep='first')"
        ),
         # Subset and ignore_index=True -> subset, keep='first', maintain_order=True
        (
            "df.drop_duplicates(subset='a', ignore_index=True)",
            "df_pl.unique(subset='a', maintain_order=True, keep='first')"
        ),
    ]
)
def test_drop_duplicates_translations(pandas_code, expected_polars):
    """Test the translation of pandas drop_duplicates to polars unique."""
    # Note: For inplace=True, the test checks the generated expression,
    # not the surrounding assignment handled by the main translator visitor.
    assert translate_test_code(pandas_code.strip()) == expected_polars.strip() 