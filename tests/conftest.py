"""Shared pytest configuration and fixtures."""
import pytest
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from loguru import logger

from pandas_to_polars_translator.logging import setup_logging

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir / "test.log")
    logger.info("Starting test session")
    yield
    logger.info("Test session completed")

@pytest.fixture
def sample_df():
    """Create a basic sample DataFrame for testing."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': [True, False, True, False, True]
    })
    logger.debug(f"Created sample DataFrame:\n{df}")
    return df

@pytest.fixture
def complex_df():
    """Create a more complex DataFrame for testing."""
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'C'] * 2,
        'subcategory': ['X', 'X', 'Y', 'Y', 'Z'] * 2,
        'value': np.random.randn(10),
        'count': np.random.randint(1, 100, 10),
        'date': pd.date_range('2023-01-01', periods=10)
    })
    logger.debug(f"Created complex DataFrame:\n{df}")
    return df

@pytest.fixture
def validation_context():
    """Create a context for validation testing."""
    def _validate_results(pandas_result, polars_result, rtol=1e-10):
        """Validate that pandas and polars results match."""
        try:
            if isinstance(pandas_result, pd.DataFrame) and isinstance(polars_result, pl.DataFrame):
                polars_as_pandas = polars_result.to_pandas()
                pd.testing.assert_frame_equal(
                    pandas_result, 
                    polars_as_pandas,
                    check_dtype=False,
                    rtol=rtol
                )
                return True
            
            elif isinstance(pandas_result, pd.Series):
                if isinstance(polars_result, pl.Series):
                    polars_values = polars_result.to_numpy()
                else:
                    polars_values = np.array([polars_result])
                return np.allclose(pandas_result.values, polars_values, rtol=rtol, equal_nan=True)
            
            elif isinstance(pandas_result, (float, int)) and isinstance(polars_result, (float, int)):
                return np.isclose(pandas_result, polars_result, rtol=rtol, equal_nan=True)
            
            return pandas_result == polars_result
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            logger.debug(f"Pandas result: {pandas_result}")
            logger.debug(f"Polars result: {polars_result}")
            return False
    
    return _validate_results 

KNOWN_IMPORT_PREFIXES = {
    "import polars as pl",
    "import polars.selectors as cs"
}

def strip_import_lines(code: str) -> str:
    """Remove known import lines if present at the start, handling multiples."""
    logger.debug(f"Stripping imports from code:\n{code}")
    lines = code.splitlines()
    stripped_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not any(line == prefix for prefix in KNOWN_IMPORT_PREFIXES):
            stripped_lines.append(line)
            
    result = '\n'.join(stripped_lines)
    logger.debug(f"After stripping imports:\n{result}")
    return result

def normalize_quotes(s: str) -> str:
    """Normalize string quotes to single quotes for comparison.
    
    Args:
        s: String to normalize
        
    Returns:
        String with normalized quotes
    """
    return s.replace('"', "'")

@pytest.fixture
def assert_translation():
    """Fixture to assert translation equality while ignoring quote style."""
    def _assert(translated: str, expected: str):
        # Strip potential import line before normalizing and comparing
        assert normalize_quotes(strip_import_lines(translated)) == normalize_quotes(expected)
    return _assert 