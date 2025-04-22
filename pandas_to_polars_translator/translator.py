"""Core translator module for converting pandas code to polars."""
import ast
from typing import Any, Dict, Optional, Set, Union, List
from enum import Enum

from .errors import UnsupportedPandasUsageError, TranslationError
from .mapping import function_maps, method_maps


class PandasToPolarsTransformer(ast.NodeTransformer):
    """AST transformer that converts pandas operations to polars operations."""
    
    def __init__(self):
        super().__init__()
        self.dataframe_vars: Set[str] = {'df'}  # Track DataFrame variables
        self.needs_polars_import = False
        self.needs_selector_import = False
        
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit a name node and transform DataFrame variable names."""
        if node.id in self.dataframe_vars:
            return ast.Name(id=f"{node.id}_pl", ctx=node.ctx)
        return node
        
    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Visit a function call node and transform pandas functions to polars."""
        # Handle pandas module function calls (e.g., pd.read_csv)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {'pd', 'pandas'}:
                return self._transform_pandas_function(node)
                
        # Handle DataFrame method calls (e.g., df.head())
        if isinstance(node.func, ast.Attribute):
            return self._transform_dataframe_method(node)
            
        return self.generic_visit(node)
        
    def _transform_pandas_function(self, node: ast.Call) -> ast.AST:
        """Transform a pandas function call to its polars equivalent."""
        func_name = node.func.attr
        
        # Check if function is supported
        if function_maps.is_special_function(func_name):
            raise UnsupportedPandasUsageError(f"Pandas function '{func_name}' is not yet supported")
            
        translation = function_maps.PANDAS_FUNCTION_TRANSLATIONS.get(func_name)
        if not translation:
            raise UnsupportedPandasUsageError(f"Pandas function '{func_name}' has no direct polars equivalent")
            
        # Transform the function call
        self.needs_polars_import = True
        new_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=translation.module, ctx=ast.Load()),
                attr=translation.polars_function,
                ctx=ast.Load()
            ),
            args=node.args,
            keywords=node.keywords
        )
            
        # Map argument names if needed
        if translation.argument_map:
            new_keywords = []
            for kw in node.keywords:
                if kw.arg in translation.argument_map:
                    if translation.argument_map[kw.arg] is not None:  # None means skip this argument
                        new_keywords.append(ast.keyword(
                            arg=translation.argument_map[kw.arg],
                            value=kw.value
                        ))
                else:
                    new_keywords.append(kw)
            new_node.keywords = new_keywords
            
        return new_node
        
    def _transform_dataframe_method(self, node: ast.Call) -> ast.AST:
        """Transform a DataFrame method call to its polars equivalent."""
        if not isinstance(node.func.value, ast.Name):
            return self.generic_visit(node)
            
        var_name = node.func.value.id
        method_name = node.func.attr
        
        # Handle string methods
        if hasattr(node.func.value, 'attr') and node.func.value.attr == 'str':
            return self._transform_string_method(node)
            
        # Check if method is supported
        if method_maps.is_special_method(method_name):
            raise UnsupportedPandasUsageError(f"Pandas method '{method_name}' is not yet supported")
            
        translation = method_maps.get_method_translation(method_name)
        if not translation:
            raise UnsupportedPandasUsageError(f"Pandas method '{method_name}' has no direct polars equivalent")
            
        # Add to tracked DataFrame variables
        self.dataframe_vars.add(var_name)

        # Set flags for imports
        self.needs_polars_import = True
        if translation.requires_selector:
            self.needs_selector_import = True

        # Handle method chaining if present
        if translation.method_chain:
            args_dict = self._args_to_dict(node)
            chain = translation.method_chain(node.args, args_dict)
            if chain:
                current_node = ast.Name(id=f"{var_name}_pl", ctx=ast.Load())
                for method, args, kwargs in chain:
                    current_node = ast.Call(
                        func=ast.Attribute(
                            value=current_node,
                            attr=method,
                            ctx=ast.Load()
                        ),
                        args=args,
                        keywords=[ast.keyword(arg=k, value=self._convert_arg_value(v)) 
                                for k, v in kwargs.items() if v is not None]
                    )
                return current_node

        # Transform the method call using direct polars method syntax
        new_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=f"{var_name}_pl", ctx=ast.Load()),
                attr=translation.polars_method,
                ctx=ast.Load()
            ),
            args=node.args,
            keywords=node.keywords
        )

        # Map argument names if needed
        if translation.argument_map:
            new_keywords = []
            for kw in node.keywords:
                if kw.arg in translation.argument_map:
                    if translation.argument_map[kw.arg] is not None:  # None means skip this argument
                        new_keywords.append(ast.keyword(
                            arg=translation.argument_map[kw.arg],
                            value=kw.value
                        ))
                else:
                    new_keywords.append(kw)
            new_node.keywords = new_keywords
            
        return new_node
        
    def _transform_string_method(self, node: ast.Call) -> ast.AST:
        """Transform a string method call to its polars equivalent."""
        method_name = node.func.attr
        translation = method_maps.get_method_translation(method_name, is_string_method=True)
        
        if not translation:
            raise UnsupportedPandasUsageError(f"Pandas string method '{method_name}' is not supported")

        # Set flag as Polars string methods use 'pl'
        self.needs_polars_import = True

        # Use polars native syntax: df.select(pl.col('col').str.method())
        # Construct the inner pl.col('col_name').str.<polars_method>(...) call
        inner_str_call = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute( # Access the 'str' namespace
                    value=ast.Call( # Create pl.col('col_name')
                        func=ast.Attribute(
                            value=ast.Name(id='pl', ctx=ast.Load()),
                            attr='col',
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=node.func.value.value.id)], # Use the original column name
                        keywords=[]
                    ),
                    attr='str',
                    ctx=ast.Load()
                ),
                attr=translation.polars_method, # The actual Polars string method
                ctx=ast.Load()
            ),
            args=node.args, # Pass original arguments
            keywords=node.keywords # Pass original keyword arguments
        )

        # Wrap the string call in df_pl.select(...) or similar context if needed
        # For now, assume it might be part of a select or with_columns,
        # but return the expression directly. The caller context should handle it.
        # Example: If original was df['A'].str.upper(), this returns pl.col('A').str.to_uppercase()
        # It needs to be placed within a df_pl.select(...) or df_pl.with_columns(...) externally.
        # TODO: Revisit how to handle the context (select vs with_columns)
        # For simple df['col'].str.method() -> df_pl.select(pl.col('col').str.method()), we can do:
        new_node = ast.Call(
             func=ast.Attribute(
                 value=ast.Name(id=f"{node.func.value.value.id}_pl", ctx=ast.Load()), # df_pl
                 attr='select', # .select()
                 ctx=ast.Load()
             ),
             args=[inner_str_call], # [pl.col(...).str.method(...)]
             keywords=[]
        )
        # Note: This assumes a transformation like df['col'].str.method() -> df_pl.select(pl.col('col').str.method())
        # More complex chains might require different handling.

        # Handle argument transformations if needed (similar to other methods)
        if translation.transform_args:
            args_dict = self._args_to_dict(node)
            transformed_args, transformed_kwargs = translation.transform_args(inner_str_call.args, args_dict) # Pass args from inner_str_call
            inner_str_call.args = transformed_args
            inner_str_call.keywords = [ast.keyword(arg=k, value=self._convert_arg_value(v))
                                    for k, v in transformed_kwargs.items()]

        elif translation.argument_map:
            new_keywords = []
            for kw in node.keywords: # Check original node's keywords
                if kw.arg in translation.argument_map:
                    mapped_arg = translation.argument_map[kw.arg]
                    if mapped_arg is not None:
                        new_keywords.append(ast.keyword(
                            arg=mapped_arg,
                            value=kw.value
                        ))
                else:
                     new_keywords.append(kw)
            inner_str_call.keywords = new_keywords # Apply mapped keywords to inner_str_call

        # For now, returning the select(...) structure. Consider if just inner_str_call is better sometimes.
        return new_node

    @staticmethod
    def _args_to_dict(node: ast.Call) -> Dict[str, Any]:
        """Convert function arguments to a dictionary."""
        args_dict = {}
        # Add positional arguments
        for i, arg in enumerate(node.args):
            args_dict[f"arg{i}"] = arg
        # Add keyword arguments
        for keyword in node.keywords:
            args_dict[keyword.arg] = keyword.value
        return args_dict
        
    @staticmethod
    def _convert_arg_value(value: Any) -> ast.AST:
        """Convert a Python value to an AST node."""
        if isinstance(value, ast.AST):
            return value
        elif isinstance(value, (list, tuple)):
            return ast.List(elts=[ast.Constant(value=x) for x in value], ctx=ast.Load())
        else:
            return ast.Constant(value=value) 