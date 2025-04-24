"""Mapping of pandas DataFrame/Series methods to polars equivalents."""
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import ast
from pd2pl.logging import logger

from .method_categories import MethodCategory, ChainableMethodTranslation

def _transform_sort_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Transform sort_values arguments to polars sort parameters.
    
    Handles:
    - Column specifications (string, list, or expressions)
    - Ascending/descending orders
    - NA position
    
    Moves column specifications to args for more idiomatic Polars code.
    """
    sort_args = []
    sort_kwargs = {}
    
    # Handle column specification
    if 'by' in kwargs:
        columns = kwargs['by']
    elif args:
        columns = args[0]
    else:
        columns = []
        
    # Normalize columns to handle single column and list cases
    if isinstance(columns, ast.List):
        # AST List node
        if len(columns.elts) == 1:
            sort_args = [columns.elts[0]]  # Single element list -> single value
        elif len(columns.elts) == 0:
            sort_args = [ast.List(elts=[], ctx=ast.Load())]  # Empty list -> keep as empty list
        else:
            sort_args = [columns]  # Multiple elements -> keep as list
    elif isinstance(columns, ast.Constant):
        # Single string constant
        sort_args = [columns]
    else:
        # Other AST node types (expressions, etc.)
        sort_args = [columns]
    
    # Handle ascending/descending
    if 'ascending' in kwargs:
        ascending = kwargs['ascending']
        if isinstance(ascending, ast.List):
            # For multiple columns, invert each boolean in the list
            sort_kwargs['descending'] = ast.List(
                elts=[
                    ast.Constant(value=not elt.value)  # Create boolean constant with inverted value
                    for elt in ascending.elts
                ],
                ctx=ast.Load()
            )
        elif isinstance(ascending, ast.Constant):
            # For single value, just invert the boolean
            sort_kwargs['descending'] = ast.Constant(value=not ascending.value)
    
    # Handle na_position
    if 'na_position' in kwargs:
        na_position = kwargs['na_position']
        if isinstance(na_position, ast.Constant):
            # 'last' -> True, 'first' -> False
            sort_kwargs['nulls_last'] = ast.Constant(value=na_position.value == 'last')
    
    return [('sort', sort_args, sort_kwargs)]

def _transform_drop_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Transform drop arguments to polars drop parameters.
    
    Handles:
    - Column specifications (string, list, or expressions)
    - Columns parameter
    - Axis parameter (ignored if columns)
    - Strict parameter
    """
    drop_args = []
    drop_kwargs = {}
    
    # Handle columns parameter first
    if 'columns' in kwargs:
        columns = kwargs['columns']
        if isinstance(columns, ast.List):
            drop_args = [columns]  # Keep list as is
        else:
            drop_args = [columns]  # Single column
    # Handle positional args if no columns parameter
    elif args:
        drop_args = args  # Keep all positional args as is
    
    # Handle strict parameter
    if 'strict' in kwargs:
        drop_kwargs['strict'] = kwargs['strict']
    
    return [('drop', drop_args, drop_kwargs)]

def _transform_sample_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Transform sample arguments to polars sample parameters.

    Ensures a consistent keyword argument output order:
    n/fraction -> seed -> shuffle -> with_replacement
    """
    sample_args = []
    # Use a temporary dict or separate checks to facilitate ordering
    temp_kwargs = {}

    # Determine primary arg (n or fraction)
    if 'n' in kwargs:
        temp_kwargs['n'] = kwargs['n']
    elif 'frac' in kwargs:
        temp_kwargs['fraction'] = kwargs['frac']
    else:
        # Apply default n=1 if neither was specified
        temp_kwargs['n'] = ast.Constant(value=1)

    # Check for other args from input
    has_seed = 'random_state' in kwargs
    is_replace_true = ('replace' in kwargs and
                       isinstance(kwargs['replace'], ast.Constant) and
                       kwargs['replace'].value is True)

    # Build the final dict in the desired order
    ordered_kwargs = {}

    # 1. Add n or fraction
    if 'n' in temp_kwargs:
        ordered_kwargs['n'] = temp_kwargs['n']
    elif 'fraction' in temp_kwargs:
        ordered_kwargs['fraction'] = temp_kwargs['fraction']

    # 2. Add seed if present
    if has_seed:
        ordered_kwargs['seed'] = kwargs['random_state']

    # 3. Add shuffle=True if seed was present
    if has_seed:
        ordered_kwargs['shuffle'] = ast.Constant(value=True)

    # 4. Add with_replacement=True if replace was True
    if is_replace_true:
        # Use the original True constant node
        ordered_kwargs['with_replacement'] = kwargs['replace']

    return [('sample', sample_args, ordered_kwargs)]

def _transform_drop_duplicates_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Transform drop_duplicates arguments to polars unique parameters.
    
    Ensures consistent keyword argument order:
    1. subset (if present)
    2. maintain_order
    3. keep
    """
    unique_args = []
    ordered_kwargs = {}  # Use OrderedDict pattern by adding keys in specific order
    
    logger.debug(f"drop_duplicates input kwargs: {kwargs}")
    
    # 1. Handle subset first (if present)
    if 'subset' in kwargs:
        ordered_kwargs['subset'] = kwargs['subset']
    
    # 2. Always add maintain_order=True
    ordered_kwargs['maintain_order'] = ast.Constant(value=True)
    
    # 3. Handle keep parameter last
    if 'keep' in kwargs:
        keep_value = kwargs['keep']
        if isinstance(keep_value, ast.Constant):
            if keep_value.value is False:
                ordered_kwargs['keep'] = ast.Constant(value='none')
            else:
                ordered_kwargs['keep'] = keep_value
    else:
        ordered_kwargs['keep'] = ast.Constant(value='first')
    
    logger.debug(f"drop_duplicates output kwargs: {ordered_kwargs}")
        
    return [('unique', unique_args, ordered_kwargs)]

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
    'rename': ChainableMethodTranslation(
        polars_method='rename',
        category=MethodCategory.BASIC,
        doc='Rename columns using a mapping or a function'
    ),
    'drop': ChainableMethodTranslation(
        polars_method='drop',
        category=MethodCategory.TRANSFORM,
        argument_map={
            'axis': None,        # Drop axis parameter
            'index': None,       # Drop index parameter
            'inplace': None,     # Drop inplace parameter
            'columns': None,     # Will be handled in method_chain
        },
        method_chain=_transform_drop_chain,
        doc='Remove columns from the DataFrame'
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
        argument_map={
            'subset': None,        # Handled by chain
            'keep': None,         # Handled by chain
            'inplace': None,      # Drop inplace parameter
            'ignore_index': None, # Drop ignore_index parameter
        },
        method_chain=_transform_drop_duplicates_chain,
        doc='Remove duplicate rows while maintaining order'
    ),
    'sample': ChainableMethodTranslation(
        polars_method='sample',
        category=MethodCategory.BASIC,
        argument_map={
            'n': 'n',                  # Pass 'n' through if not handled by default logic
            'frac': None,              # Handled by chain, drop original
            'replace': None,           # Handled by chain, drop original
            'weights': None,           # Ignore weights
            'random_state': None,      # Handled by chain, drop original
            'axis': None,              # Ignore axis
            'ignore_index': None,      # Ignore ignore_index
        },
        method_chain=_transform_sample_chain,
        doc='Sample rows from the DataFrame'
    ),
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