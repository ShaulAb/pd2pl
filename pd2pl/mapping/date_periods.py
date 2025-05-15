"""
DEPRECATED: This module is deprecated and will be removed in a future release.
Its functionality has been moved to pd2pl.datetime_utils subpackage.
Use pd2pl.datetime_utils.calculator instead.
"""

import warnings
warnings.warn(
    "The pd2pl.mapping.date_periods module is deprecated and will be removed in a future release. "
    "Use pd2pl.datetime_utils.calculator instead.",
    DeprecationWarning,
    stacklevel=2
)

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import Union, Dict, Callable

def calculate_daily_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for daily frequency."""
    return start_date + timedelta(days=periods-1)

def calculate_hourly_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for hourly frequency."""
    # For hourly frequency, we want to preserve the exact time without seconds
    result = start_date + timedelta(hours=periods-1)
    if isinstance(result, datetime):
        return result.replace(second=0, microsecond=0)
    return result

def calculate_weekly_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for weekly frequency."""
    return start_date + timedelta(weeks=periods-1)

def calculate_monthly_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for monthly frequency."""
    # For monthly frequency, we want the first day of the month after periods-1 months
    result = start_date.replace(day=1) + relativedelta(months=periods-1)
    return result

def calculate_month_end_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for month-end frequency."""
    # For month-end frequency, we want the last day of the month after periods-1 months
    result = start_date.replace(day=1) + relativedelta(months=periods)
    return result - timedelta(days=1)

def calculate_quarterly_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for quarterly frequency."""
    # For quarterly frequency, we want the last day of the quarter after periods-1 quarters
    result = start_date.replace(day=1)
    # Move to the start of the current quarter
    month = ((result.month - 1) // 3) * 3 + 1
    result = result.replace(month=month)
    # Add periods-1 quarters
    result = result + relativedelta(months=(periods-1)*3)
    # Move to the end of the quarter
    result = result + relativedelta(months=3) - timedelta(days=1)
    return result

def calculate_quarter_start_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for quarter-start frequency."""
    # For quarter-start frequency, we want the first day of the quarter after periods quarters
    result = start_date.replace(day=1)
    # Move to the start of the current quarter
    month = ((result.month - 1) // 3) * 3 + 1
    result = result.replace(month=month)
    # Add periods*3 months to get to the start of the next quarter
    result = result + relativedelta(months=periods*3)
    return result

def calculate_yearly_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for yearly frequency."""
    # For yearly frequency, we want the first day of the year after periods-1 years
    result = start_date.replace(month=1, day=1) + relativedelta(years=periods-1)
    return result

def calculate_year_start_periods(start_date: Union[date, datetime], periods: int) -> Union[date, datetime]:
    """Calculate end date for year-start frequency."""
    # For year-start frequency, we want the first day of the year after periods-1 years
    result = start_date.replace(month=1, day=1) + relativedelta(years=periods-1)
    return result

# Mapping from frequency to period calculation function
PERIOD_CALCULATORS: Dict[str, Callable[[Union[date, datetime], int], Union[date, datetime]]] = {
    'D': calculate_daily_periods,
    'H': calculate_hourly_periods,
    'W': calculate_weekly_periods,
    'M': calculate_monthly_periods,
    'ME': calculate_month_end_periods,
    'Q': calculate_quarterly_periods,
    'QS': calculate_quarter_start_periods,
    'Y': calculate_yearly_periods,
    'YS': calculate_year_start_periods,
}

def calculate_period_end_date(start_date: Union[date, datetime], periods: int, freq: str) -> Union[date, datetime]:
    """Calculate end date based on start date, periods and frequency."""
    if not periods:
        return start_date
    
    calculator = PERIOD_CALCULATORS.get(freq)
    if calculator is None:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    return calculator(start_date, periods) 