"""Date parsing and normalization utilities for pandas to polars translation."""

from datetime import date, datetime
from typing import Union

def parse_date_string(date_str: str) -> date:
    """
    Parse a date string into a date object.
    
    Args:
        date_str: A string representing a date in various formats
        
    Returns:
        A date object
        
    Raises:
        ValueError: If the date string is invalid or in an unsupported format
    """
    # Try ISO format first (YYYY-MM-DD)
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        pass
        
    # Try DD/MM/YYYY
    try:
        return datetime.strptime(date_str, '%d/%m/%Y').date()
    except ValueError:
        pass
        
    # Try YYYY/MM/DD
    try:
        return datetime.strptime(date_str, '%Y/%m/%d').date()
    except ValueError:
        pass
        
    raise ValueError(f"Could not parse date string: {date_str}")

def normalize_date_arg(date_arg: Union[str, date, datetime]) -> date:
    """
    Normalize a date argument to a date object.
    
    Args:
        date_arg: A date argument that can be a string, date, or datetime object
        
    Returns:
        A date object
        
    Raises:
        ValueError: If the argument cannot be converted to a date
    """
    if isinstance(date_arg, date):
        return date_arg
    elif isinstance(date_arg, datetime):
        # Convert datetime to date by extracting year, month, day
        return date(date_arg.year, date_arg.month, date_arg.day)
    elif isinstance(date_arg, str):
        return parse_date_string(date_arg)
    else:
        raise ValueError(f"Unsupported date argument type: {type(date_arg)}") 