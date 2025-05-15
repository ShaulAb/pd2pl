"""Tests for date parsing and normalization functionality."""

import pytest
from datetime import date, datetime
from pd2pl.datetime_utils.parser import parse_date_string, normalize_date_arg

def test_parse_date_string_iso():
    """Test parsing of ISO format date strings."""
    assert parse_date_string('2023-01-01') == date(2023, 1, 1)
    assert parse_date_string('2023-12-31') == date(2023, 12, 31)

def test_parse_date_string_common_formats():
    """Test parsing of common date string formats."""
    assert parse_date_string('01/01/2023') == date(2023, 1, 1)
    assert parse_date_string('31/12/2023') == date(2023, 12, 31)
    assert parse_date_string('2023/01/01') == date(2023, 1, 1)

def test_parse_date_string_invalid():
    """Test parsing of invalid date strings."""
    with pytest.raises(ValueError):
        parse_date_string('invalid')
    with pytest.raises(ValueError):
        parse_date_string('2023-13-01')  # Invalid month
    with pytest.raises(ValueError):
        parse_date_string('2023-01-32')  # Invalid day

def test_normalize_date_arg_date():
    """Test normalization of date objects."""
    test_date = date(2023, 1, 1)
    assert normalize_date_arg(test_date) == test_date

# def test_normalize_date_arg_datetime():
#     """Test normalization of datetime objects."""
#     test_datetime = datetime(2023, 1, 1, 12, 0)
#     assert normalize_date_arg(test_datetime) == date(2023, 1, 1)

def test_normalize_date_arg_string():
    """Test normalization of date strings."""
    assert normalize_date_arg('2023-01-01') == date(2023, 1, 1)
    assert normalize_date_arg('01/01/2023') == date(2023, 1, 1)

def test_normalize_date_arg_invalid():
    """Test normalization of invalid date arguments."""
    with pytest.raises(ValueError):
        normalize_date_arg(123)  # Invalid type
    with pytest.raises(ValueError):
        normalize_date_arg('invalid')  # Invalid string 