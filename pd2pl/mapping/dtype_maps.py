"""Centralized mapping of pandas dtypes to Polars dtypes.

Supported dtypes:
- category -> pl.Categorical
- int, int8, int16, int32, int64, Int8, Int16, Int32, Int64 -> pl.Int8, pl.Int16, pl.Int32, pl.Int64
- float, float16, float32, float64 -> pl.Float32, pl.Float64
- bool, boolean -> pl.Boolean
"""

import ast

# Map pandas dtype string/numpy dtype to Polars dtype AST expression
_DTYPE_MAP = {
    # Categorical
    'category': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Categorical', ctx=ast.Load()),
    # Integers
    'int': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int64', ctx=ast.Load()),
    'int8': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int8', ctx=ast.Load()),
    'int16': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int16', ctx=ast.Load()),
    'int32': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int32', ctx=ast.Load()),
    'int64': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int64', ctx=ast.Load()),
    'Int8': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int8', ctx=ast.Load()),
    'Int16': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int16', ctx=ast.Load()),
    'Int32': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int32', ctx=ast.Load()),
    'Int64': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Int64', ctx=ast.Load()),
    # Floats
    'float': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Float64', ctx=ast.Load()),
    'float16': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Float32', ctx=ast.Load()),
    'float32': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Float32', ctx=ast.Load()),
    'float64': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Float64', ctx=ast.Load()),
    # Booleans
    'bool': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Boolean', ctx=ast.Load()),
    'boolean': ast.Attribute(value=ast.Name(id='pl', ctx=ast.Load()), attr='Boolean', ctx=ast.Load()),
}

def to_polars_dtype(pandas_dtype):
    """
    Map a pandas dtype string (or numpy dtype) to a Polars dtype AST node.
    Returns None if no mapping is found.
    """
    if isinstance(pandas_dtype, ast.Constant):
        key = pandas_dtype.value
    elif isinstance(pandas_dtype, str):
        key = pandas_dtype
    else:
        return None
    return _DTYPE_MAP.get(key, None) 