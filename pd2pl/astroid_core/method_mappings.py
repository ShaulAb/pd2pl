"""Safe method name mappings extracted from legacy mapping data.

This module contains only the basic method name mappings and argument mappings
that are safe to reuse. It does NOT contain transformation functions or complex logic.
"""

# Basic method name mappings: pandas_method -> polars_method
BASIC_METHOD_MAPPINGS = {
    # Aggregation methods
    'mean': 'mean',
    'sum': 'sum', 
    'max': 'max',
    'min': 'min',
    'count': 'count',
    
    # Transform methods
    'groupby': 'group_by',
    'sort_values': 'sort',
    'drop_duplicates': 'unique',
    'fillna': 'fill_null',
    
    # Basic methods
    'head': 'head',
    'tail': 'tail',
    'describe': 'describe',
    'rename': 'rename',
    'drop': 'drop',
    
    # Reshape methods
    'melt': 'unpivot',
    'pivot': 'pivot',
}

# Aggregation methods that need special pl.all() wrapper
AGGREGATION_METHODS = {
    'mean', 'sum', 'max', 'min', 'count', 'std', 'var'
}

def get_polars_method_name(pandas_method: str) -> str:
    """Get the polars equivalent of a pandas method name.
    
    Args:
        pandas_method: The pandas method name
        
    Returns:
        str: The polars method name, or the original if no mapping exists
    """
    return BASIC_METHOD_MAPPINGS.get(pandas_method, pandas_method)

def is_aggregation_method(method_name: str) -> bool:
    """Check if a method is an aggregation method that needs pl.all() wrapper.
    
    Args:
        method_name: The method name to check
        
    Returns:
        bool: True if it's an aggregation method
    """
    return method_name in AGGREGATION_METHODS 