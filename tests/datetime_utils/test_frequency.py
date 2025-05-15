"""Tests for frequency parsing and mapping functionality."""

import pytest
from pd2pl.datetime_utils.frequency import parse_frequency, map_pandas_freq_to_polars_interval

def test_parse_frequency_basic():
    """Test parsing of basic frequency strings."""
    assert parse_frequency('D') == ('D', 1)
    assert parse_frequency('W') == ('W', 1)
    assert parse_frequency('M') == ('M', 1)
    assert parse_frequency('Y') == ('Y', 1)

def test_parse_frequency_with_multiplier():
    """Test parsing of frequency strings with numeric multipliers."""
    assert parse_frequency('2D') == ('D', 2)
    assert parse_frequency('3W') == ('W', 3)
    assert parse_frequency('4M') == ('M', 4)
    assert parse_frequency('5Y') == ('Y', 5)

def test_parse_frequency_invalid():
    """Test parsing of invalid frequency strings."""
    with pytest.raises(ValueError):
        parse_frequency('invalid')
    with pytest.raises(ValueError):
        parse_frequency('')
    with pytest.raises(ValueError):
        parse_frequency('0D')

def test_map_pandas_freq_to_polars_interval_basic():
    """Test mapping of basic pandas frequencies to polars intervals."""
    assert map_pandas_freq_to_polars_interval('D') == '1d'
    assert map_pandas_freq_to_polars_interval('W') == '1w'
    assert map_pandas_freq_to_polars_interval('M') == '1mo'
    assert map_pandas_freq_to_polars_interval('Y') == '1y'

def test_map_pandas_freq_to_polars_interval_with_multiplier():
    """Test mapping of pandas frequencies with multipliers to polars intervals."""
    assert map_pandas_freq_to_polars_interval('2D') == '2d'
    assert map_pandas_freq_to_polars_interval('3W') == '3w'
    assert map_pandas_freq_to_polars_interval('4M') == '4mo'
    assert map_pandas_freq_to_polars_interval('5Y') == '5y' 