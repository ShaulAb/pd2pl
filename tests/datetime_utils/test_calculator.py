"""Tests for date calculation functionality."""

import pytest
from datetime import date, datetime, timedelta
from pd2pl.datetime_utils.calculator import add_time_interval, calculate_periods_end_date

def test_add_time_interval_days():
    """Test adding day intervals."""
    assert add_time_interval(date(2023, 1, 1), '1d') == date(2023, 1, 2)
    assert add_time_interval(date(2023, 1, 1), '5d') == date(2023, 1, 6)

def test_add_time_interval_weeks():
    """Test adding week intervals."""
    assert add_time_interval(date(2023, 1, 1), '1w') == date(2023, 1, 8)
    assert add_time_interval(date(2023, 1, 1), '2w') == date(2023, 1, 15)

def test_add_time_interval_months():
    """Test adding month intervals."""
    assert add_time_interval(date(2023, 1, 1), '1mo') == date(2023, 2, 1)
    assert add_time_interval(date(2023, 1, 31), '1mo') == date(2023, 2, 28)  # End of month handling
    assert add_time_interval(date(2023, 1, 31), '2mo') == date(2023, 3, 31)

def test_add_time_interval_quarters():
    """Test adding quarter intervals."""
    assert add_time_interval(date(2023, 1, 1), '1q') == date(2023, 4, 1)
    assert add_time_interval(date(2023, 1, 1), '2q') == date(2023, 7, 1)

def test_add_time_interval_years():
    """Test adding year intervals."""
    assert add_time_interval(date(2023, 1, 1), '1y') == date(2024, 1, 1)
    # Leap year handling
    assert add_time_interval(date(2020, 2, 29), '1y') == date(2021, 2, 28)

def test_add_time_interval_time_components():
    """Test adding time intervals to datetime objects."""
    assert add_time_interval(datetime(2023, 1, 1, 12, 0), '1h') == datetime(2023, 1, 1, 13, 0)
    assert add_time_interval(datetime(2023, 1, 1, 12, 0), '30m') == datetime(2023, 1, 1, 12, 30)
    assert add_time_interval(datetime(2023, 1, 1, 12, 0), '90s') == datetime(2023, 1, 1, 12, 1, 30)
    assert add_time_interval(datetime(2023, 1, 1, 12, 0), '500ms') == datetime(2023, 1, 1, 12, 0, 0, 500000)

def test_add_time_interval_invalid():
    """Test adding invalid intervals."""
    with pytest.raises(ValueError):
        add_time_interval(date(2023, 1, 1), 'invalid')
    with pytest.raises(ValueError):
        add_time_interval(date(2023, 1, 1), '1x')  # Invalid unit

def test_calculate_periods_end_date_days():
    """Test calculating end dates with daily frequencies."""
    assert calculate_periods_end_date(date(2023, 1, 1), 1, 'D') == date(2023, 1, 1)
    assert calculate_periods_end_date(date(2023, 1, 1), 5, 'D') == date(2023, 1, 5)
    assert calculate_periods_end_date(date(2023, 1, 1), 3, '2D') == date(2023, 1, 5)  # 2-day jumps

def test_calculate_periods_end_date_weeks():
    """Test calculating end dates with weekly frequencies."""
    assert calculate_periods_end_date(date(2023, 1, 1), 1, 'W') == date(2023, 1, 1)
    assert calculate_periods_end_date(date(2023, 1, 1), 2, 'W') == date(2023, 1, 8)
    assert calculate_periods_end_date(date(2023, 1, 1), 3, 'W') == date(2023, 1, 15)

def test_calculate_periods_end_date_months():
    """Test calculating end dates with monthly frequencies."""
    # Month-end frequencies should align to the last day of each month
    assert calculate_periods_end_date(date(2023, 1, 15), 1, 'ME') == date(2023, 1, 15)
    assert calculate_periods_end_date(date(2023, 1, 15), 2, 'ME') == date(2023, 2, 28)
    assert calculate_periods_end_date(date(2023, 1, 31), 2, 'ME') == date(2023, 2, 28)
    assert calculate_periods_end_date(date(2023, 1, 31), 3, 'ME') == date(2023, 3, 31)

def test_calculate_periods_end_date_quarters():
    """Test calculating end dates with quarterly frequencies."""
    # Quarter-end frequencies should align to the last day of Mar, Jun, Sep, Dec
    assert calculate_periods_end_date(date(2023, 1, 15), 1, 'QE') == date(2023, 1, 15)
    assert calculate_periods_end_date(date(2023, 1, 15), 2, 'QE') == date(2023, 3, 31)
    assert calculate_periods_end_date(date(2023, 1, 15), 3, 'QE') == date(2023, 6, 30)

def test_calculate_periods_end_date_years():
    """Test calculating end dates with yearly frequencies."""
    # Year-end frequencies should align to December 31
    assert calculate_periods_end_date(date(2023, 6, 15), 1, 'YE') == date(2023, 6, 15)
    assert calculate_periods_end_date(date(2023, 6, 15), 2, 'YE') == date(2023, 12, 31)
    assert calculate_periods_end_date(date(2023, 6, 15), 3, 'YE') == date(2024, 12, 31)

def test_calculate_periods_end_date_time_components():
    """Test calculating end dates with time components."""
    start_dt = datetime(2023, 1, 1, 12, 30)
    assert calculate_periods_end_date(start_dt, 1, 'H') == start_dt
    assert calculate_periods_end_date(start_dt, 3, 'H') == datetime(2023, 1, 1, 14, 30)
    assert calculate_periods_end_date(start_dt, 24, 'H') == datetime(2023, 1, 2, 11, 30)

def test_calculate_periods_end_date_invalid():
    """Test calculating end dates with invalid inputs."""
    with pytest.raises(ValueError):
        calculate_periods_end_date(date(2023, 1, 1), 0, 'D')  # Invalid periods
    with pytest.raises(ValueError):
        calculate_periods_end_date(date(2023, 1, 1), -1, 'D')  # Invalid periods
    with pytest.raises(ValueError):
        calculate_periods_end_date(date(2023, 1, 1), 5, 'X')  # Invalid frequency 