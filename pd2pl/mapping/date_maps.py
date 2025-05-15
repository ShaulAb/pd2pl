"""
DEPRECATED: This module is deprecated and will be removed in a future release.
Its functionality has been moved to pd2pl.datetime_utils subpackage.
Use pd2pl.datetime_utils.calculator and pd2pl.datetime_utils.transformers instead.
"""

import warnings
warnings.warn(
    "The pd2pl.mapping.date_maps module is deprecated and will be removed in a future release. "
    "Use pd2pl.datetime_utils subpackage instead.",
    DeprecationWarning,
    stacklevel=2
)

from typing import Dict, Optional, Union, TYPE_CHECKING
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import ast
import re
from .date_periods import calculate_period_end_date

if TYPE_CHECKING:
    from pd2pl.translator import PandasToPolarsVisitor

# Mapping from pandas frequency strings to polars interval strings
FREQ_TO_INTERVAL: Dict[str, str] = {
    # Basic frequencies
    'D': '1d',      # Daily
    'W': '1w',      # Weekly
    'M': '1mo',     # Monthly
    'Q': '1q',      # Quarterly
    'Y': '1y',      # Yearly
    'H': '1h',      # Hourly
    'T': '1m',      # Minute
    'S': '1s',      # Second
    'MS': '1mo',    # Month Start
    'QS': '1q',     # Quarter Start
    'YS': '1y',     # Year Start
    'ME': '1mo',    # Month End
    'QE': '1q',     # Quarter End
    'YE': '1y',     # Year End
}

# Mapping from pandas inclusive values to polars closed values
INCLUSIVE_TO_CLOSED: Dict[str, str] = {
    'both': 'both',
    'left': 'left',
    'right': 'right',
    'neither': 'none'
}

def calculate_end_date(start_date: Union[date, datetime], periods: int, freq: str) -> Union[date, datetime]:
    """Calculate end date based on start date, periods and frequency, aligning to period boundaries as pandas does."""
    if not periods:
        return start_date

    # Get the interval string from frequency
    interval = FREQ_TO_INTERVAL.get(freq, '1d')
    
    # Extract numeric part and unit using regex
    match = re.match(r'(\d+)([a-zA-Z]+)', interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    
    number = int(match.group(1))
    unit = match.group(2)
    
    # For period-end frequencies, align to the last day of the period
    if freq in ('M', 'ME'):
        # Month end: go to the last day of the month for each period
        result = start_date
        for _ in range(periods):
            # Go to the last day of the current month
            next_month = result.replace(day=1) + relativedelta(months=1)
            result = next_month - timedelta(days=1)
        return result
    elif freq in ('Q', 'QE'):
        # Quarter end: last day of Mar, Jun, Sep, Dec
        result = start_date
        for _ in range(periods):
            # Go to the last day of the current quarter
            month = ((result.month - 1) // 3 + 1) * 3
            year = result.year
            if month > 12:
                month -= 12
                year += 1
            # Get the first day of the next quarter
            next_quarter = date(year, month, 1) + relativedelta(months=1)
            result = next_quarter - timedelta(days=1)
        return result
    elif freq in ('Y', 'YE'):
        # Year end: December 31st
        result = start_date
        for _ in range(periods):
            next_year = result.replace(month=1, day=1) + relativedelta(years=1)
            result = next_year - timedelta(days=1)
        return result
    elif freq in ('MS',):
        # Month start: first day of the month
        result = start_date
        for _ in range(periods-1):
            result = result.replace(day=1) + relativedelta(months=1)
        return result
    elif freq in ('QS',):
        # Quarter start: first day of the quarter
        result = start_date
        for _ in range(periods-1):
            month = ((result.month - 1) // 3) * 3 + 1
            year = result.year
            result = result.replace(year=year, month=month, day=1) + relativedelta(months=3)
        return result
    elif freq in ('YS',):
        # Year start: January 1st
        result = start_date
        for _ in range(periods-1):
            result = result.replace(month=1, day=1) + relativedelta(years=1)
        return result
    # Default: use timedelta/relativedelta as before
    total_units = number * (periods - 1)
    if unit == 'd':
        return start_date + timedelta(days=total_units)
    elif unit == 'h':
        return start_date + timedelta(hours=total_units)
    elif unit == 'm':
        return start_date + timedelta(minutes=total_units)
    elif unit == 's':
        return start_date + timedelta(seconds=total_units)
    elif unit == 'w':
        return start_date + timedelta(weeks=total_units)
    elif unit == 'mo':
        return start_date + relativedelta(months=total_units)
    elif unit == 'q':
        return start_date + relativedelta(months=total_units * 3)
    elif unit == 'y':
        return start_date + relativedelta(years=total_units)
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

def create_date_ast_node(value: Union[datetime, date]) -> ast.Call:
    """Create an AST node for a date or datetime object."""
    if isinstance(value, datetime):
        return ast.Call(
            func=ast.Name(id='datetime', ctx=ast.Load()),
            args=[
                ast.Constant(value=value.year),
                ast.Constant(value=value.month),
                ast.Constant(value=value.day),
                ast.Constant(value=value.hour),
                ast.Constant(value=value.minute),
                ast.Constant(value=value.second)
            ],
            keywords=[]
        )
    else:  # date
        return ast.Call(
            func=ast.Name(id='date', ctx=ast.Load()),
            args=[
                ast.Constant(value=value.year),
                ast.Constant(value=value.month),
                ast.Constant(value=value.day)
            ],
            keywords=[]
        )

def translate_date_range(node: ast.Call, *, visitor: 'PandasToPolarsVisitor', **kwargs) -> ast.AST:
    """
    Translate pandas date_range call to polars date_range.
    
    Args:
        node: AST node for pandas date_range call
        visitor: The visitor instance for translating nested elements
        
    Returns:
        AST node for polars date_range call
    """
    # This function delegates to the datetime_utils implementation
    from pd2pl.datetime_utils.transformers import transform_date_range
    return transform_date_range(node, visitor) 