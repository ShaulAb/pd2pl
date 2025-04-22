"""Mapping of pandas functions to their polars equivalents."""
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class FunctionTranslation:
    """Represents how to translate a pandas function to polars."""
    polars_function: str
    module: str = 'pl'
    argument_map: Optional[Dict[str, str]] = None
    transform_args: Optional[Callable[[list, dict], tuple[list, dict]]] = None
    doc: str = ""

# Direct function name mappings (pd.function_name -> pl.function_name)
FUNCTION_NAME_MAP: Dict[str, str] = {
    'read_csv': 'read_csv',
    'concat': 'concat',
    'merge': 'join',  # Note: arguments need transformation
}

# Function argument transformations
def transform_merge_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Transform pandas merge arguments to polars join arguments."""
    transformed = args.copy()
    # Map common argument names
    if 'left' in transformed:
        transformed['left_df'] = transformed.pop('left')
    if 'right' in transformed:
        transformed['right_df'] = transformed.pop('right')
    return transformed

# Map of functions that need argument transformation
FUNCTION_ARG_TRANSFORMERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    'merge': transform_merge_args,
}

# Functions that require special handling beyond simple mapping
SPECIAL_FUNCTIONS = {
    'get_dummies',  # Currently unsupported
    'melt',         # Requires complex transformation
    'pivot_table',  # Requires complex transformation
}

def get_polars_function(pandas_func: str) -> Optional[str]:
    """Get the polars equivalent of a pandas function name.
    
    Args:
        pandas_func: Name of the pandas function
        
    Returns:
        Optional[str]: Name of the equivalent polars function, or None if not supported
    """
    return FUNCTION_NAME_MAP.get(pandas_func)

def get_arg_transformer(func_name: str) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Get the argument transformer for a function if it exists.
    
    Args:
        func_name: Name of the function
        
    Returns:
        Optional[Callable]: Function to transform arguments, or None if no transformation needed
    """
    return FUNCTION_ARG_TRANSFORMERS.get(func_name)

def is_special_function(func_name: str) -> bool:
    """Check if a function requires special handling.
    
    Args:
        func_name: Name of the function
        
    Returns:
        bool: True if the function requires special handling
    """
    return func_name in SPECIAL_FUNCTIONS

# Basic function translations
PANDAS_FUNCTION_TRANSLATIONS: Dict[str, FunctionTranslation] = {
    'read_csv': FunctionTranslation(
        polars_function='read_csv',
        argument_map={
            'filepath_or_buffer': 'source',
            'sep': 'separator',
            'header': 'has_header',
            'names': 'columns',
            'usecols': 'columns',
            'dtype': 'dtypes'
        },
        doc='Read a comma-separated values (csv) file'
    ),
    'concat': FunctionTranslation(
        polars_function='concat',
        argument_map={
            'objs': 'dfs',
            'axis': 'how',
        },
        transform_args=lambda args, kwargs: (
            args,
            {**kwargs, 'how': 'vertical' if kwargs.get('axis', 0) == 0 else 'horizontal'}
        ),
        doc='Concatenate DataFrames'
    ),
    'merge': FunctionTranslation(
        polars_function='join',
        argument_map={
            'left': 'other',
            'right': None,  # Will be handled specially in translator
            'how': 'how',
            'on': 'on',
            'left_on': 'left_on',
            'right_on': 'right_on'
        },
        doc='Merge DataFrames'
    )
} 