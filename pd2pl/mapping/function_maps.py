"""Mapping of pandas functions to their polars equivalents."""
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
import ast
from pd2pl.errors import TranslationError
from pd2pl.logging import logger

# Add type hinting for visitor if using explicit signature
if TYPE_CHECKING:
    from pd2pl.translator import PandasToPolarsVisitor

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

def _parse_merge_args(node: ast.Call) -> dict:
    """Basic parser for pd.merge arguments. Focuses on keywords."""
    args = {}
    # Assume first two args are left and right dfs if present positionally
    if len(node.args) >= 1:
        args['left'] = node.args[0]
    if len(node.args) >= 2:
        args['right'] = node.args[1]

    # Process keywords, overwriting positional if needed (like pandas)
    for kw in node.keywords:
        args[kw.arg] = kw.value

    # --- Set defaults for parsing --- 
    # We need defaults primarily to check if unsupported features were *explicitly* used
    args.setdefault('how', ast.Constant(value='inner'))
    args.setdefault('left_index', ast.Constant(value=False))
    args.setdefault('right_index', ast.Constant(value=False))
    # We need suffixes default to check if lsuffix was explicitly set to non-empty
    # args.setdefault('suffixes', ast.Tuple(elts=[ast.Constant(value='_x'), ast.Constant(value='_y')], ctx=ast.Load()))

    return args

# Explicitly define visitor type hint
def translate_merge(node: ast.Call, *, visitor: 'PandasToPolarsVisitor', **kwargs) -> ast.AST:
    """Translates a pd.merge(left, right, ...) call for COLUMN joins only."""
    parsed_args = _parse_merge_args(node)

    left_df_node = parsed_args.get('left')
    right_df_node = parsed_args.get('right')

    if left_df_node is None or right_df_node is None:
        raise TranslationError("Could not identify left and right DataFrames in pd.merge call.")

    # --- Check for unsupported features --- 
    left_index_node = parsed_args.get('left_index', ast.Constant(value=False))
    right_index_node = parsed_args.get('right_index', ast.Constant(value=False))
    if not isinstance(left_index_node, ast.Constant) or left_index_node.value is not False:
         raise TranslationError("Index joins in pd.merge (left_index=True) are not yet supported.")
    if not isinstance(right_index_node, ast.Constant) or right_index_node.value is not False:
         raise TranslationError("Index joins in pd.merge (right_index=True) are not yet supported.")

    # --- Handle suffixes --- (Determine rsuffix value if provided)
    suffixes_provided = 'suffixes' in parsed_args
    lsuffix = '_x' # pandas default
    rsuffix = '_y' # pandas default

    if suffixes_provided:
        suffixes_node = parsed_args['suffixes']
        if isinstance(suffixes_node, (ast.Tuple, ast.List)) and len(suffixes_node.elts) == 2:
            if isinstance(suffixes_node.elts[0], ast.Constant) and isinstance(suffixes_node.elts[0].value, str):
                lsuffix = suffixes_node.elts[0].value
            else:
                raise TranslationError("lsuffix must be a string literal.")
            if isinstance(suffixes_node.elts[1], ast.Constant) and isinstance(suffixes_node.elts[1].value, str):
                rsuffix = suffixes_node.elts[1].value # Keep the provided rsuffix
            else:
                raise TranslationError("rsuffix must be a string literal.")
        else:
            raise TranslationError("'suffixes' argument must be a tuple/list of two strings.")

        # Check for unsupported lsuffix only if suffixes were provided
        if lsuffix not in ('', '_x'): # Moved check here for clarity
            raise TranslationError(f"lsuffix ('{lsuffix}') is not supported. Only rsuffix is used by Polars join.")

    # --- Explicitly visit/transform left and right nodes ---
    transformed_left_node = visitor.visit(left_df_node)
    transformed_right_node = visitor.visit(right_df_node)

    # --- Build Polars join arguments ---
    join_kwargs_ast = {}

    # Keys (must be column joins)
    if 'on' in parsed_args:
        if 'left_on' in parsed_args or 'right_on' in parsed_args:
             raise TranslationError("Cannot specify both 'on' and 'left_on'/'right_on'.")
        join_kwargs_ast['on'] = parsed_args['on']
    elif 'left_on' in parsed_args and 'right_on' in parsed_args:
        join_kwargs_ast['left_on'] = parsed_args['left_on']
        join_kwargs_ast['right_on'] = parsed_args['right_on']
    else:
        raise TranslationError("pd.merge requires 'on' or both 'left_on' and 'right_on' for column joins.")

    # How
    how_node = parsed_args.get('how', ast.Constant(value='inner')) # Default inner
    if isinstance(how_node, ast.Constant) and isinstance(how_node.value, str):
        how_val = how_node.value
        polars_how = 'full' if how_val == 'outer' else how_val
        # Only allow subset supported by both and sensible for merge mapping
        if polars_how not in ('inner', 'left', 'right', 'full'):
            raise TranslationError(f"Unsupported 'how' value for pd.merge: {how_val}. Supported: inner, left, right, outer.")
        join_kwargs_ast['how'] = ast.Constant(value=polars_how)
    else:
        raise TranslationError("'how' argument must be a string literal (e.g., 'left', 'inner').")

    # --- Conditional Suffix --- (Reverted Logic)
    if suffixes_provided:
        # Only add suffix if 'suffixes' was originally provided
        polars_suffix = rsuffix if rsuffix not in ('', '_y') else '_right'
        join_kwargs_ast['suffix'] = ast.Constant(value=polars_suffix)

    # Validate
    if 'validate' in parsed_args:
        validate_node = parsed_args['validate']
        if not isinstance(validate_node, ast.Constant) or validate_node.value not in ('1:1', '1:m', 'm:1', 'm:m'):
            raise TranslationError(f"Invalid or non-literal 'validate' argument: {getattr(validate_node, 'value', validate_node)}")
        join_kwargs_ast['validate'] = validate_node

    # Add coalesce=True by default
    join_kwargs_ast['coalesce'] = ast.Constant(value=True)

    # Construct the join call
    join_call = ast.Call(
        func=ast.Attribute(value=transformed_left_node, attr='join', ctx=ast.Load()),
        args=[transformed_right_node],
        keywords=[ast.keyword(arg=k, value=v) for k, v in join_kwargs_ast.items()]
    )

    return join_call

# Dictionary to hold function translations
FUNCTION_TRANSLATIONS: Dict[str, Callable] = {
    'merge': translate_merge,
} 