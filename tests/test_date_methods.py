import pytest
from tests.conftest import translate_test_code
from tests._helpers import compare_frames
import pandas as pd
import polars as pl
from datetime import date, datetime

@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic date_range with start and end
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), freq='D')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval='1d')"
        ),
        # String date input with periods
        (
            "pd.date_range(start='20130101', periods=6)",
            "pl.date_range(start='20130101', periods=6, interval='1d')"
        ),
        # date_range with periods
        (
            "pd.date_range(start=date(2022, 1, 1), periods=5, freq='D')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 5), interval='1d')"
        ),
        # date_range with inclusive='left'
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), freq='D', inclusive='left')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval='1d', closed='left')"
        ),
        # date_range with inclusive='right'
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), freq='D', inclusive='right')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval='1d', closed='right')"
        ),
        # date_range with inclusive='both'
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), freq='D', inclusive='both')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval='1d', closed='both')"
        ),
        # date_range with inclusive='neither'
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), freq='D', inclusive='neither')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval='1d', closed='none')"
        ),
        # Different frequencies
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), freq='H')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval='1h')"
        ),
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 3, 1), freq='M')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 3, 1), interval='1mo')"
        ),
        (
            "pd.date_range(start=date(2022, 1, 1), end=date(2022, 1, 31), freq='W')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 31), interval='1w')"
        ),
        # datetime inputs
        (
            "pd.date_range(start=datetime(2022, 1, 1, 12, 0), end=datetime(2022, 1, 2, 12, 0), freq='H')",
            "pl.date_range(start=datetime(2022, 1, 1, 12, 0), end=datetime(2022, 1, 2, 12, 0), interval='1h')"
        ),
        # Additional test cases for different time frames
        # Monthly periods
        (
            "pd.date_range(start=date(2022, 1, 1), periods=3, freq='M')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 3, 1), interval='1mo')"
        ),
        # Weekly periods
        (
            "pd.date_range(start=date(2022, 1, 1), periods=4, freq='W')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 22), interval='1w')"
        ),
        # Hourly periods
        (
            "pd.date_range(start=datetime(2022, 1, 1, 12, 0), periods=6, freq='H')",
            "pl.date_range(start=datetime(2022, 1, 1, 12, 0), end=datetime(2022, 1, 1, 17, 0, 0), interval='1h')"
        ),
        # Quarterly periods
        (
            "pd.date_range(start=date(2022, 1, 1), periods=2, freq='Q')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 6, 30), interval='1q')"
        ),
        # Yearly periods
        (
            "pd.date_range(start=date(2022, 1, 1), periods=3, freq='Y')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2024, 1, 1), interval='1y')"
        ),
        # Edge cases
        # Single period
        (
            "pd.date_range(start=date(2022, 1, 1), periods=1, freq='D')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 1), interval='1d')"
        ),
        # Month end frequency
        (
            "pd.date_range(start=date(2022, 1, 1), periods=3, freq='ME')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 3, 31), interval='1mo')"
        ),
        # Quarter start frequency
        (
            "pd.date_range(start=date(2022, 1, 1), periods=2, freq='QS')",
            "pl.date_range(start=date(2022, 1, 1), end=date(2022, 7, 1), interval='1q')"
        ),
    ]
)
def test_date_range_translations(pandas_code, expected_polars):
    """
    Test the translation of pandas date_range to polars date_range.
    
    This test only checks that the translation from pandas to polars syntax is correct.
    It does not validate that the output of the functions are identical, as there are 
    some differences in behavior between pandas and polars date range functions.
    """
    # Translation check
    translated = translate_test_code(pandas_code.strip())
    assert translated == expected_polars.strip()

class TestComplexDateExpressions:
    """Test date_range with complex expressions.
    
    This class will be implemented later when we handle expression translations.
    Complex expressions include:
    - Variable references
    - Function calls
    - etc.
    """
    pass 