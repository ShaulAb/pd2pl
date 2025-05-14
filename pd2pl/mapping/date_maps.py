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
    """Translate pandas date_range to polars date_range."""
    # Extract arguments from the pandas date_range call
    args = {}
    for keyword in node.keywords:
        if keyword.arg == 'start':
            args['start'] = visitor.visit(keyword.value)
        elif keyword.arg == 'end':
            args['end'] = visitor.visit(keyword.value)
        elif keyword.arg == 'periods':
            args['periods'] = visitor.visit(keyword.value)
        elif keyword.arg == 'freq':
            args['freq'] = keyword.value  # Don't visit this as we need the raw value
        elif keyword.arg == 'inclusive':
            args['inclusive'] = keyword.value  # Don't visit this as we need the raw value

    # Build keywords for polars date_range call
    keywords = []
    
    # Handle start
    if 'start' in args:
        keywords.append(ast.keyword(arg='start', value=args['start']))
    
    # Handle end/periods
    if 'end' in args:
        keywords.append(ast.keyword(arg='end', value=args['end']))
    elif 'periods' in args and 'start' in args and 'freq' in args:
        # Calculate end date from periods
        if (isinstance(args['freq'], ast.Constant) and 
            isinstance(args['freq'].value, str) and
            isinstance(args['periods'], ast.Constant) and 
            isinstance(args['periods'].value, int)):
            
            # Get the start value from the AST
            start_value = None
            if isinstance(args['start'], ast.Call):
                if isinstance(args['start'].func, ast.Name):
                    if args['start'].func.id == 'date':
                        start_value = date(
                            args['start'].args[0].value,
                            args['start'].args[1].value,
                            args['start'].args[2].value
                        )
                    elif args['start'].func.id == 'datetime':
                        start_value = datetime(
                            args['start'].args[0].value,
                            args['start'].args[1].value,
                            args['start'].args[2].value,
                            args['start'].args[3].value if len(args['start'].args) > 3 else 0,
                            args['start'].args[4].value if len(args['start'].args) > 4 else 0,
                            args['start'].args[5].value if len(args['start'].args) > 5 else 0
                        )
            
            if start_value is not None:
                # Calculate the end date using the new period calculation module
                end_date = calculate_period_end_date(
                    start_value,
                    args['periods'].value,
                    args['freq'].value
                )
                # Create AST node for the end date
                end_node = create_date_ast_node(end_date)
                keywords.append(ast.keyword(arg='end', value=end_node))
    
    # Handle freq/interval
    if 'freq' in args and isinstance(args['freq'], ast.Constant) and isinstance(args['freq'].value, str):
        interval = FREQ_TO_INTERVAL.get(args['freq'].value, '1d')
        keywords.append(ast.keyword(
            arg='interval',
            value=ast.Constant(value=interval)
        ))
    
    # Handle inclusive/closed
    if 'inclusive' in args and isinstance(args['inclusive'], ast.Constant) and isinstance(args['inclusive'].value, str):
        closed = INCLUSIVE_TO_CLOSED.get(args['inclusive'].value, 'both')
        keywords.append(ast.keyword(
            arg='closed',
            value=ast.Constant(value=closed)
        ))
    
    # Always add eager=True for pandas compatibility
    keywords.append(ast.keyword(
        arg='eager',
        value=ast.Constant(value=True)
    ))
    
    # Construct the final call as an AST node
    date_range_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='pl', ctx=ast.Load()),
            attr='date_range',
            ctx=ast.Load()
        ),
        args=[],
        keywords=keywords
    )
    
    return date_range_call 