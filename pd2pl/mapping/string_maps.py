"""
Mapping system for pandas string operations to polars equivalents.
"""
from typing import Dict, Any, Callable, List, Tuple, Optional

from pd2pl.mapping.method_categories import MethodCategory

# Direct name mappings (when method names match or are simple renames)
STRING_METHOD_DIRECT_MAP = {
    'split': 'split',
    'replace': 'replace',
    'contains': 'contains',
    'startswith': 'starts_with',
    'endswith': 'ends_with',
    'lower': 'to_lowercase',
    'upper': 'to_uppercase',
    'strip': 'strip_chars',
    'lstrip': 'strip_chars_start',
    'rstrip': 'strip_chars_end',
    'len': 'len_chars',
    'extract': 'extract',
}

# Complex transformations that need parameter remapping
def transform_split(args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Transform pandas Series.str.split to polars Series.str.split"""
    new_args = list(args)
    new_kwargs = dict()
    
    # Handle parameter differences
    # Rename 'pat' to 'by' if present
    if 'pat' in kwargs:
        new_kwargs['by'] = kwargs['pat']
    elif len(args) > 0:
        # Keep positional arg as is
        new_args = args
    
    # Copy other supported parameters
    if 'n' in kwargs:
        new_kwargs['n'] = kwargs['n']
    
    # Handle 'expand' parameter which has no direct equivalent
    # We ignore 'expand' as it's not supported in Polars
        
    return new_args, new_kwargs

def transform_replace(args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Transform pandas Series.str.replace to polars Series.str.replace"""
    new_args = list(args)
    new_kwargs = dict()
    
    # Keep positional arguments (pat, repl)
    if len(args) > 0:
        new_args = args
    
    # Handle 'n' parameter (limit in pandas) if present
    if 'n' in kwargs:
        new_kwargs['n'] = kwargs['n']
    
    # We don't copy 'regex' parameter as Polars uses regex by default
    # We don't copy 'case' parameter as it's not supported in Polars
    
    return new_args, new_kwargs

def transform_extract(args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Transform pandas Series.str.extract to polars Series.str.extract"""
    new_args = list(args)
    new_kwargs = dict()
    
    # Keep positional args (pattern)
    if len(args) > 0:
        new_args = args
    
    # We don't copy 'expand' or 'flags' parameters as they're not supported in Polars
    
    return new_args, new_kwargs

# Register the transformations
STRING_METHOD_TRANSFORMATIONS = {
    'split': transform_split,
    'replace': transform_replace,
    'extract': transform_extract,
}

# Information for string methods for integration with the method mapping system
from pd2pl.mapping.method_categories import ChainableMethodTranslation

STRING_METHODS_INFO = {
    # Direct mappings with minimal transformation
    'contains': ChainableMethodTranslation(
        polars_method='contains',
        category=MethodCategory.STRING_METHODS,
        doc='Test if pattern or regex is contained within string'
    ),
    'len': ChainableMethodTranslation(
        polars_method='len_chars',
        category=MethodCategory.STRING_METHODS,
        doc='Return the length of each string as the number of characters'
    ),
    'lower': ChainableMethodTranslation(
        polars_method='to_lowercase',
        category=MethodCategory.STRING_METHODS, 
        doc='Convert strings to lowercase'
    ),
    'upper': ChainableMethodTranslation(
        polars_method='to_uppercase',
        category=MethodCategory.STRING_METHODS,
        doc='Convert strings to uppercase'
    ),
    'strip': ChainableMethodTranslation(
        polars_method='strip_chars',
        category=MethodCategory.STRING_METHODS,
        doc='Remove leading and trailing whitespace'
    ),
    'lstrip': ChainableMethodTranslation(
        polars_method='strip_chars_start',
        category=MethodCategory.STRING_METHODS,
        doc='Remove leading whitespace'
    ),
    'rstrip': ChainableMethodTranslation(
        polars_method='strip_chars_end', 
        category=MethodCategory.STRING_METHODS,
        doc='Remove trailing whitespace'
    ),
    'startswith': ChainableMethodTranslation(
        polars_method='starts_with',
        category=MethodCategory.STRING_METHODS,
        doc='Test if string starts with pattern'
    ),
    'endswith': ChainableMethodTranslation(
        polars_method='ends_with',
        category=MethodCategory.STRING_METHODS,
        doc='Test if string ends with pattern'
    ),
    
    # Methods with parameter transformations
    'split': ChainableMethodTranslation(
        polars_method='split',
        category=MethodCategory.STRING_METHODS,
        method_chain=lambda args, kwargs: [('split', *transform_split(args, kwargs))],
        doc='Split strings on separator/delimiter'
    ),
    'replace': ChainableMethodTranslation(
        polars_method='replace',
        category=MethodCategory.STRING_METHODS,
        method_chain=lambda args, kwargs: [('replace', *transform_replace(args, kwargs))],
        doc='Replace occurrences of pattern/regex with replacement string'
    ),
    'extract': ChainableMethodTranslation(
        polars_method='extract',
        category=MethodCategory.STRING_METHODS,
        method_chain=lambda args, kwargs: [('extract', *transform_extract(args, kwargs))],
        doc='Extract capture groups from string as columns using regex'
    ),
}

# Export the maps for use in method_maps.py
__all__ = ['STRING_METHOD_DIRECT_MAP', 'STRING_METHOD_TRANSFORMATIONS', 'STRING_METHODS_INFO'] 