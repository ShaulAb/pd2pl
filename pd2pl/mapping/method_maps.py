"""Mapping of pandas DataFrame/Series methods to polars equivalents."""
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import ast

from .method_categories import MethodCategory, ChainableMethodTranslation

def _transform_sort_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Transform sort_values arguments to polars sort parameters.
    
    Handles:
    - Column specifications (string, list, or expressions)
    - Ascending/descending orders
    - NA position
    
    Moves column specifications to args for more idiomatic Polars code.
    """
    sort_args = []  # Store columns in args
    sort_kwargs = {}
    
    # Handle column specification
    if 'by' in kwargs:
        sort_args = [kwargs['by']]  # Move 'by' to args
    elif args:
        sort_args = [args[0]]  # Use the first positional arg
    
    # Handle ascending/descending
    if 'ascending' in kwargs:
        ascending = kwargs['ascending']
        if isinstance(ascending, (list, tuple)):
            # For multiple columns, invert each boolean in the list
            sort_kwargs['descending'] = [not a for a in ascending]
        else:
            # For single value, just invert the boolean
            sort_kwargs['descending'] = not ascending
    
    # Handle na_position
    if 'na_position' in kwargs:
        # 'last' -> True, 'first' -> False
        sort_kwargs['nulls_last'] = kwargs['na_position'] == 'last'
    
    # If sort_args contains a single-element list, unpack it
    if len(sort_args) == 1 and isinstance(sort_args[0], list) and len(sort_args[0]) == 1:
        sort_args = [sort_args[0][0]]
    
    return [('sort', sort_args, sort_kwargs)]

# Basic method translations
DATAFRAME_METHOD_TRANSLATIONS: Dict[str, ChainableMethodTranslation] = {
    'head': ChainableMethodTranslation(
        polars_method='head',
        category=MethodCategory.BASIC,
        doc='Returns first n rows'
    ),
    'tail': ChainableMethodTranslation(
        polars_method='tail',
        category=MethodCategory.BASIC,
        doc='Returns last n rows'
    ),
    'describe': ChainableMethodTranslation(
        polars_method='describe',
        category=MethodCategory.BASIC,
        doc='Generate descriptive statistics'
    ),
    'fillna': ChainableMethodTranslation(
        polars_method='fill_null',
        category=MethodCategory.TRANSFORM,
        doc='Fill null values with a constant value'
    ),
    'mean': ChainableMethodTranslation(
        polars_method='mean',
        category=MethodCategory.AGGREGATION,
        argument_map={'numeric_only': None},
        method_chain=lambda args, kwargs: ([
            ('select', [ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='cs', ctx=ast.Load()),
                    attr='numeric',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )], {}),
            ('mean', [], {})
        ] if kwargs.get('numeric_only') else None),
        requires_selector=True,
        doc='Compute mean of numeric columns'
    ),
    'sum': ChainableMethodTranslation(
        polars_method='sum',
        category=MethodCategory.AGGREGATION,
        argument_map={'numeric_only': None},
        method_chain=lambda args, kwargs: ([
            ('select', [ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='cs', ctx=ast.Load()),
                    attr='numeric',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )], {}),
            ('sum', [], {})
        ] if kwargs.get('numeric_only') else None),
        requires_selector=True,
        doc='Compute sum of columns'
    ),
    'max': ChainableMethodTranslation(
        polars_method='max',
        category=MethodCategory.AGGREGATION,
        doc='Compute maximum of columns'
    ),
    'min': ChainableMethodTranslation(
        polars_method='min',
        category=MethodCategory.AGGREGATION,
        doc='Compute minimum of columns'
    ),
    'count': ChainableMethodTranslation(
        polars_method='count',
        category=MethodCategory.AGGREGATION,
        doc='Count non-null values'
    ),
    'groupby': ChainableMethodTranslation(
        polars_method='group_by',
        category=MethodCategory.TRANSFORM,
        method_chain=lambda args, kwargs: ([
            ('group_by', args, {})
        ]),
        doc='Group DataFrame by columns'
    ),
    'sort_values': ChainableMethodTranslation(
        polars_method='sort',
        category=MethodCategory.TRANSFORM,
        argument_map={
            'by': None,                    # Drop 'by' from mapping
            'ascending': 'descending',     # Will be transformed in method_chain
            'na_position': 'nulls_last',   # Will be transformed in method_chain
            'axis': None,                  # Drop axis parameter
            'inplace': None,               # Drop inplace parameter
            'kind': None,                  # Drop kind parameter
            'ignore_index': None,          # Drop ignore_index parameter
        },
        method_chain=_transform_sort_chain,
        doc='Sort DataFrame by values'
    ),
    'reset_index': ChainableMethodTranslation(
        polars_method='reset_index',
        category=MethodCategory.TRANSFORM,
        doc='Reset DataFrame index'
    ),
    'drop_duplicates': ChainableMethodTranslation(
        polars_method='unique',
        category=MethodCategory.TRANSFORM,
        argument_map={'subset': 'subset', 'keep': 'keep'},
        doc='Remove duplicate rows'
    )
}

# String method translations
STRING_METHOD_TRANSLATIONS: Dict[str, ChainableMethodTranslation] = {
    'contains': ChainableMethodTranslation(
        polars_method='contains',
        category=MethodCategory.BASIC,
        doc='Test if pattern or regex is contained within string'
    ),
    'len': ChainableMethodTranslation(
        polars_method='len_chars',
        category=MethodCategory.BASIC,
        doc='Return the length of each string as the number of characters'
    ),
    'replace': ChainableMethodTranslation(
        polars_method='replace',
        category=MethodCategory.BASIC,
        doc='Replace occurrences of pattern/regex with some other string'
    ),
    'lower': ChainableMethodTranslation(
        polars_method='to_lowercase',
        category=MethodCategory.BASIC,
        doc='Convert strings to lowercase'
    ),
    'upper': ChainableMethodTranslation(
        polars_method='to_uppercase',
        category=MethodCategory.BASIC,
        doc='Convert strings to uppercase'
    ),
    'strip': ChainableMethodTranslation(
        polars_method='strip',
        category=MethodCategory.BASIC,
        doc='Remove leading and trailing whitespace'
    )
}

# Methods that require special handling beyond simple mapping
SPECIAL_METHODS = {
    'iloc',         # Requires complex transformation
    'loc',          # Requires complex transformation
    'resample',     # Requires special time handling
    # 'expanding' is implicitly handled via chains like cum_sum
}

def get_method_translation(method_name: str, is_string_method: bool = False) -> Optional[ChainableMethodTranslation]:
    """Get the translation information for a pandas method.
    
    Args:
        method_name: Name of the pandas method
        is_string_method: Whether this is a string method (df.str.*)
        
    Returns:
        Optional[ChainableMethodTranslation]: Translation information if available
    """
    if is_string_method:
        return STRING_METHOD_TRANSLATIONS.get(method_name)
    return DATAFRAME_METHOD_TRANSLATIONS.get(method_name)

def is_special_method(method_name: str) -> bool:
    """Check if a method requires special handling.
    
    Args:
        method_name: Name of the method
        
    Returns:
        bool: True if the method requires special handling
    """
    # We check for rolling/expanding chains directly in the translator now
    return method_name in SPECIAL_METHODS 