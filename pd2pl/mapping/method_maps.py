"""Mapping of pandas DataFrame/Series methods to polars equivalents."""
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import ast
from pd2pl.logging import logger
from pd2pl.errors import TranslationError

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

# Add supported aggfunc mapping for pivot_table
SUPPORTED_PIVOT_AGG = {
    'mean', 'sum', 'min', 'max', 'median', 'count', 'first', 'last'
}

def _transform_pivot_table_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """
    Transform pivot_table arguments to polars pivot parameters.

    Handles:
    - Mapping index, columns, values.
    - Mapping 'aggfunc' -> 'aggregate_function', defaulting to 'mean'.
    - Handling 'fill_value' by chaining '.fill_null()'.
    - Raises errors for unsupported parameters/aggfuncs.
    """
    pivot_kwargs = {}
    fill_value_node = None
    method_steps = []

    # Check for unsupported args first
    unsupported_args = {'margins', 'dropna', 'sort'} # Sort is ignored, others error
    for arg in unsupported_args:
        if arg in kwargs and arg != 'sort': # Allow sort but ignore it
             raise TranslationError(f"Unsupported pivot_table parameter: {arg}")
             # Or: logger.warning(f"Ignoring unsupported pivot_table parameter: {arg}")

    # Map core arguments
    for pd_arg, pl_arg in [('index', 'index'), ('columns', 'on'), ('values', 'values')]:
        if pd_arg in kwargs:
            pivot_kwargs[pl_arg] = kwargs[pd_arg]
        elif pd_arg == 'values' and pd_arg not in kwargs:
             raise TranslationError("pivot_table translation requires explicit 'values' parameter.")

    # Handle aggfunc
    aggfunc_node = kwargs.get('aggfunc', ast.Constant(value='mean')) # Default to mean
    if isinstance(aggfunc_node, ast.Constant) and isinstance(aggfunc_node.value, str):
        aggfunc_str = aggfunc_node.value
        aggfunc_out = 'len' if aggfunc_str == 'count' else aggfunc_str
        if aggfunc_str in SUPPORTED_PIVOT_AGG:
             pivot_kwargs['aggregate_function'] = ast.Constant(value=aggfunc_out)
        else:
             raise TranslationError(f"Unsupported pivot_table aggfunc string: '{aggfunc_str}'")
    else:
         raise TranslationError("Unsupported pivot_table aggfunc: Only string values like 'mean', 'sum', 'count' are supported.")

    # Handle fill_value
    if 'fill_value' in kwargs:
        fill_value_node = kwargs['fill_value']

    # Add pivot step
    method_steps.append(('pivot', [], pivot_kwargs))

    # Add fill_null step if needed
    if fill_value_node is not None:
        method_steps.append(('fill_null', [fill_value_node], {}))

    return method_steps

# Registry of supported aggregation functions for groupby
SUPPORTED_GROUPBY_AGGS = {
    'sum': 'sum',
    'mean': 'mean',
    'min': 'min',
    'max': 'max',
    'count': 'count',
    'median': 'median',
    'first': 'first',
    'last': 'last',
}

# Mapping from aggregation function to Polars selector attribute
AGG_TO_SELECTOR = {
    'sum': 'numeric',
    'mean': 'numeric',
    'median': 'numeric',
    'std': 'numeric',
    'var': 'numeric',
    'min': 'numeric_or_string',  # min/max can work on strings too
    'max': 'numeric_or_string',
    'count': 'all',
    'first': 'all',
    'last': 'all',
    'nunique': 'all',
}

# Helper to build selector AST
import ast

def selector_ast(selector):
    if selector == 'numeric':
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='cs', ctx=ast.Load()), attr='numeric', ctx=ast.Load()),
            args=[], keywords=[])
    elif selector == 'string':
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='cs', ctx=ast.Load()), attr='string', ctx=ast.Load()),
            args=[], keywords=[])
    elif selector == 'all':
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='cs', ctx=ast.Load()), attr='all', ctx=ast.Load()),
            args=[], keywords=[])
    elif selector == 'numeric_or_string':
        # cs.numeric() + cs.string()
        return ast.BinOp(
            left=ast.Call(
                func=ast.Attribute(value=ast.Name(id='cs', ctx=ast.Load()), attr='numeric', ctx=ast.Load()),
                args=[], keywords=[]),
            op=ast.Add(),
            right=ast.Call(
                func=ast.Attribute(value=ast.Name(id='cs', ctx=ast.Load()), attr='string', ctx=ast.Load()),
                args=[], keywords=[]))
    else:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='cs', ctx=ast.Load()), attr='all', ctx=ast.Load()),
            args=[], keywords=[])

def _transform_groupby_agg_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """
    Transform groupby.agg arguments to polars groupby.agg parameters.
    Handles:
    - Dictionary style: {col: agg} or {col: [agg1, agg2, ...]}
    - Tuple/keyword style: newcol=(col, agg)
    - Simple aggregations (e.g., .sum(), .mean())
    Dtype-aware: only generate aggs for columns compatible with the agg, using selectors.
    """
    method_steps = []
    agg_exprs = []
    # Helper to build .sum().alias('foo')
    def agg_with_alias(col, agg, alias):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='pl', ctx=ast.Load()),
                                attr='col',
                                ctx=ast.Load()
                            ),
                            args=[ast.Constant(value=col)],
                            keywords=[]
                        ),
                        attr=agg,
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                ),
                attr='alias',
                ctx=ast.Load()
            ),
            args=[ast.Constant(value=alias)],
            keywords=[]
        )
    # Dict-style: df.groupby(...).agg({'val1': 'sum', ...} or {'val1': ['sum', 'mean'], ...})
    if args and isinstance(args[0], ast.Dict):
        dict_arg = args[0]
        for col_node, agg_node in zip(dict_arg.keys, dict_arg.values):
            if isinstance(col_node, ast.Constant):
                col = col_node.value
                # Single aggregation as string
                if isinstance(agg_node, ast.Constant):
                    agg = agg_node.value
                    polars_agg = SUPPORTED_GROUPBY_AGGS.get(agg)
                    selector = AGG_TO_SELECTOR.get(agg, 'all')
                    if polars_agg:
                        agg_exprs.append(agg_with_alias(col, polars_agg, f"{col}_{agg}"))
                    else:
                        raise TranslationError(f"Unsupported aggregation function: {agg}")
                # Multiple aggregations as list
                elif isinstance(agg_node, ast.List):
                    for elt in agg_node.elts:
                        if isinstance(elt, ast.Constant):
                            agg = elt.value
                            polars_agg = SUPPORTED_GROUPBY_AGGS.get(agg)
                            selector = AGG_TO_SELECTOR.get(agg, 'all')
                            if polars_agg:
                                agg_exprs.append(agg_with_alias(col, polars_agg, f"{col}_{agg}"))
                            else:
                                raise TranslationError(f"Unsupported aggregation function: {agg}")
                        else:
                            raise TranslationError("Aggregation list must contain only string constants")
                else:
                    raise TranslationError("Aggregation dictionary values must be a string or list of strings")
            else:
                raise TranslationError("Aggregation dictionary keys must be string constants")
    # Tuple/keyword style: df.groupby(...).agg(newcol=(col, agg), ...)
    elif kwargs:
        for newcol, val in kwargs.items():
            if isinstance(val, ast.Tuple) and len(val.elts) == 2:
                col_node, agg_node = val.elts
                if isinstance(col_node, ast.Constant) and isinstance(agg_node, ast.Constant):
                    col = col_node.value
                    agg = agg_node.value
                    polars_agg = SUPPORTED_GROUPBY_AGGS.get(agg)
                    selector = AGG_TO_SELECTOR.get(agg, 'all')
                    if polars_agg:
                        agg_exprs.append(agg_with_alias(col, polars_agg, newcol))
                    else:
                        raise TranslationError(f"Unsupported aggregation function: {agg}")
                else:
                    raise TranslationError("Unsupported tuple aggregation format")
            else:
                raise TranslationError("Unsupported tuple aggregation format")
    # Fallback: treat as simple aggregation (e.g., .sum(), .mean())
    elif args and isinstance(args[0], ast.List):
        # Already a list of expressions (advanced usage)
        agg_exprs = args[0].elts
    else:
        # .sum(), .mean(), etc. on groupby
        # Use selectors for dtype-aware translation
        if args:
            agg_method = args[0].attr if isinstance(args[0], ast.Attribute) else None
        else:
            agg_method = None
        if agg_method in AGG_TO_SELECTOR:
            selector = AGG_TO_SELECTOR[agg_method]
            agg_exprs = [ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='pl', ctx=ast.Load()),
                            attr='col',
                            ctx=ast.Load()
                        ),
                        args=[selector_ast(selector)],
                        keywords=[]
                    ),
                    attr=agg_method,
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )]
        else:
            raise TranslationError("Unsupported .agg() format after groupby")
    method_steps.append(('agg', [ast.List(elts=agg_exprs, ctx=ast.Load())], {}))
    return method_steps

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
    'melt': ChainableMethodTranslation(
        polars_method='unpivot',
        category=MethodCategory.RESHAPE,
        argument_map={
            'id_vars': 'index',
            'value_vars': 'on',
            'var_name': 'variable_name',
            'value_name': 'value_name',
            'col_level': None,      # Ignore col_level
            'ignore_index': None,   # Ignore ignore_index
        },
        doc='Unpivot DataFrame from wide to long format (similar to melt)'
    ),
    'pivot': ChainableMethodTranslation(
        polars_method='pivot',
        category=MethodCategory.RESHAPE,
        argument_map={
            'index': 'index',
            'columns': 'on',
            'values': 'values', # TODO: Handle case where values is None (needs schema info)
        },
        # No method_chain needed for now, relies on explicit 'values'
        doc='Reshape data (produce a "pivot" table) based on column values.'
            ' Requires explicit "values" argument. Uses default Polars aggregation ("first").'
    ),
    'pivot_table': ChainableMethodTranslation(
        polars_method='pivot', # Base method is pivot
        category=MethodCategory.RESHAPE,
        argument_map={
            # Arguments handled by the chain function
            'index': None,
            'columns': None,
            'values': None,
            'aggfunc': None,
            'fill_value': None,
            # Ignored/Unsupported arguments
            'margins': None,
            'dropna': None,
            'sort': None, # Explicitly ignore
        },
        method_chain=_transform_pivot_table_chain,
        doc='Reshape data using aggregation (similar to pivot_table).' 
            ' Limited support for aggfunc (strings only) and parameters.' 
            ' Handles fill_value via chained fill_null.'
    ),
    'agg': ChainableMethodTranslation(
        polars_method='agg',
        category=MethodCategory.AGGREGATION,
        method_chain=_transform_groupby_agg_chain,
        doc='Aggregate using one or more operations over the specified axis.'
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