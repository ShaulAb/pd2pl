"""AST transformers for date-related pandas functions."""

import ast
from datetime import date, datetime
from typing import Dict, Optional, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pd2pl.translator import PandasToPolarsVisitor

from .frequency import map_pandas_freq_to_polars_interval
from .calculator import calculate_periods_end_date
from .parser import normalize_date_arg

# Mapping from pandas inclusive values to polars closed values
INCLUSIVE_TO_CLOSED: Dict[str, str] = {
    'both': 'both',
    'left': 'left',
    'right': 'right',
    'neither': 'none'
}

def create_date_ast_node(value: Union[datetime, date]) -> ast.Call:
    """
    Create an AST node for a date or datetime object.
    
    Args:
        value: A date or datetime object
        
    Returns:
        An AST Call node representing the date or datetime constructor
    """
    if isinstance(value, datetime):
        # Create argument list with all non-zero components
        args = [
            ast.Constant(value=value.year),
            ast.Constant(value=value.month),
            ast.Constant(value=value.day),
        ]
        
        # Add time components only if needed
        if value.hour != 0 or value.minute != 0 or value.second != 0 or value.microsecond != 0:
            args.extend([
                ast.Constant(value=value.hour),
                ast.Constant(value=value.minute),
                ast.Constant(value=value.second),
            ])
            # Only add microsecond if non-zero
            if value.microsecond != 0:
                args.append(ast.Constant(value=value.microsecond))
                
        return ast.Call(
            func=ast.Name(id='datetime', ctx=ast.Load()),
            args=args,
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

def transform_date_range(node: ast.Call, visitor: 'PandasToPolarsVisitor') -> ast.Call:
    """
    Transform pandas date_range call to polars date_range.
    
    Args:
        node: AST node for pandas date_range call
        visitor: The visitor instance for translating nested elements
        
    Returns:
        AST node for polars date_range call
    """
    # Extract arguments from the pandas date_range call
    args = {}
    for keyword in node.keywords:
        if keyword.arg in ('start', 'end', 'periods'):
            args[keyword.arg] = visitor.visit(keyword.value)
        elif keyword.arg == 'freq':
            # Don't visit this as we need to preserve the raw string value
            args['freq'] = keyword.value
        elif keyword.arg == 'inclusive':
            # Don't visit this as we need to preserve the raw string value
            args['inclusive'] = keyword.value
        # Ignore other arguments that aren't supported in polars

    # Build keywords for polars date_range call
    keywords = []
    
    # Handle start (required)
    if 'start' in args:
        keywords.append(ast.keyword(arg='start', value=args['start']))
    
    # Handle end/periods - prioritize end if both are specified
    if 'end' in args:
        keywords.append(ast.keyword(arg='end', value=args['end']))
    elif 'periods' in args and 'start' in args:
        # If we have a frequency and can extract constant values, calculate end date
        if 'freq' in args and isinstance(args['freq'], ast.Constant) and isinstance(args['periods'], ast.Constant):
            freq_value = args['freq'].value
            periods_value = args['periods'].value
            
            # Try to extract a concrete date from the start node
            start_value = None
            if isinstance(args['start'], ast.Call) and isinstance(args['start'].func, ast.Name):
                if args['start'].func.id == 'date' and all(isinstance(arg, ast.Constant) for arg in args['start'].args):
                    year, month, day = [arg.value for arg in args['start'].args]
                    start_value = date(year, month, day)
                elif args['start'].func.id == 'datetime' and all(isinstance(arg, ast.Constant) for arg in args['start'].args):
                    args_values = [arg.value for arg in args['start'].args]
                    if len(args_values) >= 3:
                        year, month, day = args_values[:3]
                        hour = args_values[3] if len(args_values) > 3 else 0
                        minute = args_values[4] if len(args_values) > 4 else 0
                        second = args_values[5] if len(args_values) > 5 else 0
                        microsecond = args_values[6] if len(args_values) > 6 else 0
                        start_value = datetime(year, month, day, hour, minute, second, microsecond)
            
            if start_value is not None and periods_value > 0:
                # Calculate the end date
                end_date = calculate_periods_end_date(start_value, periods_value, freq_value)
                end_node = create_date_ast_node(end_date)
                keywords.append(ast.keyword(arg='end', value=end_node))
            else:
                # If we can't compute the end date, pass periods directly
                keywords.append(ast.keyword(arg='periods', value=args['periods']))
        else:
            # For non-constant or missing frequency, pass periods directly
            keywords.append(ast.keyword(arg='periods', value=args['periods']))
    
    # Handle freq/interval
    if 'freq' in args:
        if isinstance(args['freq'], ast.Constant) and isinstance(args['freq'].value, str):
            # Map pandas frequency to polars interval
            interval = map_pandas_freq_to_polars_interval(args['freq'].value)
            keywords.append(ast.keyword(
                arg='interval',
                value=ast.Constant(value=interval)
            ))
        else:
            # For dynamic frequency, create runtime mapping call
            map_call = ast.Call(
                func=ast.Name(id='map_pandas_freq_to_polars_interval', ctx=ast.Load()),
                args=[args['freq']],
                keywords=[]
            )
            keywords.append(ast.keyword(arg='interval', value=map_call))
    else:
        # Default interval is '1d'
        keywords.append(ast.keyword(
            arg='interval',
            value=ast.Constant(value='1d')
        ))
    
    # Handle inclusive/closed
    if 'inclusive' in args:
        if isinstance(args['inclusive'], ast.Constant) and isinstance(args['inclusive'].value, str):
            closed = INCLUSIVE_TO_CLOSED.get(args['inclusive'].value, 'both')
            keywords.append(ast.keyword(
                arg='closed',
                value=ast.Constant(value=closed)
            ))
        else:
            # For dynamic inclusive, create a lookup
            inclusive_mapping = ast.Dict(
                keys=[ast.Constant(value=k) for k in INCLUSIVE_TO_CLOSED.keys()],
                values=[ast.Constant(value=v) for v in INCLUSIVE_TO_CLOSED.values()]
            )
            # Use dict.get with default value 'both'
            get_call = ast.Call(
                func=ast.Attribute(
                    value=inclusive_mapping,
                    attr='get',
                    ctx=ast.Load()
                ),
                args=[args['inclusive'], ast.Constant(value='both')],
                keywords=[]
            )
            keywords.append(ast.keyword(arg='closed', value=get_call))
    
    # Construct the final polars date_range call
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='pl', ctx=ast.Load()),
            attr='date_range',
            ctx=ast.Load()
        ),
        args=[],
        keywords=keywords
    ) 