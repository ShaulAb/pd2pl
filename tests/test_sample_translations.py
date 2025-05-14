# tests/test_sample_translations.py
import pytest

from tests.conftest import translate_test_code
@pytest.mark.parametrize(
    "pandas_code, expected_polars",
    [
        # Sample by n (integer) - omit shuffle=False
        (
            "df.sample(n=3)",
            "df_pl.sample(n=3)"
        ),
        # Sample by frac (float) - omit shuffle=False
        (
            "df.sample(frac=0.5)",
            "df_pl.sample(fraction=0.5)"
        ),
        # Sample with replacement - omit shuffle=False, keep with_replacement=True
        (
            "df.sample(n=5, replace=True)",
            "df_pl.sample(n=5, with_replacement=True)"
        ),
        # Sample with random_state - keep seed and shuffle=True
        (
            "df.sample(n=3, random_state=42)",
            "df_pl.sample(n=3, seed=42, shuffle=True)" # Order: n, seed, shuffle
        ),
        # Sample with frac and random_state - keep seed and shuffle=True
        (
            "df.sample(frac=0.2, random_state=123)",
            "df_pl.sample(fraction=0.2, seed=123, shuffle=True)" # Order: fraction, seed, shuffle
        ),
        # Sample with replacement and random_state - keep seed, shuffle=True, with_replacement=True
        (
            "df.sample(n=5, replace=True, random_state=1)",
            "df_pl.sample(n=5, seed=1, shuffle=True, with_replacement=True)" # Order: n, seed, shuffle, with_replacement
        ),
        # Ignore axis parameter - omit shuffle=False
        (
            "df.sample(n=3, axis=0)",
             "df_pl.sample(n=3)"
        ),
        # Ignore axis parameter - omit shuffle=False
        (
            "df.sample(n=3, axis=1)",
             "df_pl.sample(n=3)"
        ),
         # Ignore axis parameter - omit shuffle=False
        (
            "df.sample(n=3, axis='columns')",
             "df_pl.sample(n=3)"
        ),
        # Ignore ignore_index parameter - omit shuffle=False
        (
            "df.sample(n=3, ignore_index=True)",
            "df_pl.sample(n=3)"
        ),
        # Ignore weights parameter - omit shuffle=False
        (
            "df.sample(n=3, weights='my_weights_column')",
            "df_pl.sample(n=3)"
        ),
        # Default n=1 - omit shuffle=False
        (
            "df.sample()",
            "df_pl.sample(n=1)"
        ),
    ]
)
def test_sample_translations(pandas_code, expected_polars):
    """Test the translation of pandas sample to polars sample."""
    assert translate_test_code(pandas_code.strip()) == expected_polars.strip() 