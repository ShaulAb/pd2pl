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
        # Handle module function calls first (e.g., pd.read_csv)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {'pd', 'pandas'}:
                return self._transform_pandas_function(node)

        # Handle DataFrame method calls
        if isinstance(node.func, ast.Attribute):
            # Check for string methods (df['col'].str.method())
            if (isinstance(node.func.value, ast.Attribute) and
                node.func.value.attr == 'str'):
                return self._transform_string_method(node)

            # Check for window operations (rolling/expanding)
            if (isinstance(node.func.value, ast.Call) and
                isinstance(node.func.value.func, ast.Attribute) and
                node.func.value.func.attr in {'rolling', 'expanding'}):
                return self._transform_window_operation(node)

            # Check for groupby chains
            if (isinstance(node.func.value, ast.Call) and
                isinstance(node.func.value.func, ast.Attribute) and
                node.func.value.func.attr == 'groupby'):
                return self._transform_groupby_agg(node)

            # Check for other method calls (df.method())
            if isinstance(node.func.value, ast.Name):
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
        
    def _transform_groupby_agg(self, node: ast.Call) -> ast.AST:
        """Transform a groupby aggregation chain (e.g., df.groupby().mean())."""
        agg_method = node.func.attr
        groupby_call_node = node.func.value

        # First, transform the inner groupby call
        transformed_groupby = self.visit_Call(groupby_call_node)

        # Determine the Polars aggregation expression
        agg_exprs = []
        if agg_method == 'mean':
            # df.groupby(...).mean() -> df_pl.groupby(...).agg(pl.all().mean())
            self.needs_polars_import = True
            agg_exprs = [ast.Call(
                func=ast.Attribute(
                    value=ast.Call(  # pl.all()
                        func=ast.Attribute(
                            value=ast.Name(id='pl', ctx=ast.Load()),
                            attr='all',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    ),
                    attr='mean',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )]
        elif agg_method == 'agg':
            # df.groupby(...).agg({...}) -> df_pl.groupby(...).agg([...])
            self.needs_polars_import = True
            if node.args and isinstance(node.args[0], ast.Dict):
                dict_arg = node.args[0]
                for col_name_node, agg_func_node in zip(dict_arg.keys, dict_arg.values):
                    if isinstance(col_name_node, ast.Constant) and isinstance(agg_func_node, ast.Constant):
                        col_name = col_name_node.value
                        agg_func_str = agg_func_node.value
                        # Create pl.col('col_name').agg_func()
                        agg_exprs.append(ast.Call(
                            func=ast.Attribute(
                                value=ast.Call(  # pl.col('col_name')
                                    func=ast.Attribute(
                                        value=ast.Name(id='pl', ctx=ast.Load()),
                                        attr='col',
                                        ctx=ast.Load()
                                    ),
                                    args=[ast.Constant(value=col_name)],
                                    keywords=[]
                                ),
                                attr=agg_func_str,  # Use the pandas agg func name directly for simple cases
                                ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[]
                        ))
                    else:
                        raise TranslationError("Unsupported aggregation dictionary format")
            else:
                raise TranslationError("Unsupported .agg() format after groupby")
        else:
            # Handle other potential aggregations like .sum(), .max(), etc.
            # Default: Assume simple aggregation pl.all().agg_method()
            self.needs_polars_import = True
            agg_exprs = [ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='pl', ctx=ast.Load()),
                            attr='all',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    ),
                    attr=agg_method,
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )]

        # Construct the final Polars node: df_pl.groupby(...).agg([...])
        final_node = ast.Call(
            func=ast.Attribute(
                value=transformed_groupby,  # The result of df_pl.groupby(...)
                attr='agg',
                ctx=ast.Load()
            ),
            args=[ast.List(elts=agg_exprs, ctx=ast.Load()) if len(agg_exprs) > 1 else agg_exprs[0]],
            keywords=[]
        )
        return final_node
        
    def _transform_dataframe_method(self, node: ast.Call) -> ast.AST:
        """Transform a DataFrame method call to its polars equivalent."""
        # Note: This method now primarily handles *non-chained* calls on a DataFrame variable,
        # or chains where the base is a simple variable (e.g., df.head()).
        # Complex chains like groupby().agg() are handled by _transform_groupby_agg.
        # String chains (df.col.str.method()) are handled by _transform_string_method.
        
        # Ensure the base of the call is a simple variable name
        if not isinstance(node.func.value, ast.Name):
            # Could be a subscript (df['col'].method()) or other complex base.
            # We might need specific handlers for these later.
            return self.generic_visit(node) # Fallback for now

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

        # Get the column name from the chain
        # df['col'].str.method() -> node.func.value.value is the Subscript
        # df.col.str.method() -> node.func.value.value is the Name
        if isinstance(node.func.value.value, ast.Subscript):
            col_name = node.func.value.value.slice.value
        else:
            col_name = node.func.value.value.id

        self.needs_polars_import = True

        # Create pl.col('col_name').str.method() expression
        inner_expr = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(  # Access the 'str' namespace
                    value=ast.Call(  # Create pl.col('col_name')
                        func=ast.Attribute(
                            value=ast.Name(id='pl', ctx=ast.Load()),
                            attr='col',
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=col_name)],
                        keywords=[]
                    ),
                    attr='str',
                    ctx=ast.Load()
                ),
                attr=translation.polars_method,
                ctx=ast.Load()
            ),
            args=node.args,
            keywords=node.keywords
        )

        # Always wrap in df_pl.select(...)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='df_pl', ctx=ast.Load()),
                attr='select',
                ctx=ast.Load()
            ),
            args=[inner_expr],
            keywords=[]
        )

    def _transform_window_operation(self, node: ast.Call) -> ast.AST:
        """Transform window operations (rolling/expanding) to their polars equivalents."""
        window_method = node.func.attr  # mean, sum, etc.
        window_call = node.func.value   # The rolling() or expanding() call
        window_type = window_call.func.attr  # 'rolling' or 'expanding'
        
        # Get the base column reference
        if isinstance(window_call.func.value, ast.Subscript):
            col_name = window_call.func.value.slice.value
        else:
            raise TranslationError("Window operations must be called on a column")

        self.needs_polars_import = True
        
        # Create the appropriate Polars expression
        if window_type == 'rolling':
            # Get window size from the rolling() call
            window_size = None
            for kw in window_call.keywords:
                if kw.arg == 'window':
                    if isinstance(kw.value, ast.Constant):
                        window_size = kw.value.value
                    break
            if window_size is None and window_call.args:
                window_size = window_call.args[0].value if isinstance(window_call.args[0], ast.Constant) else None
            if window_size is None:
                raise TranslationError("Could not determine window size for rolling operation")

            # Create pl.col('col').rolling_mean(window_size=N)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='df_pl', ctx=ast.Load()),
                    attr='select',
                    ctx=ast.Load()
                ),
                args=[
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='pl', ctx=ast.Load()),
                                    attr='col',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Constant(value=col_name)],
                                keywords=[]
                            ),
                            attr=f'rolling_{window_method}',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[ast.keyword(arg='window_size', value=ast.Constant(value=window_size))]
                    )
                ],
                keywords=[]
            )
        elif window_type == 'expanding':
            # Map expanding operations to their Polars equivalents
            method_map = {
                'sum': 'cum_sum',
                'mean': 'cum_mean',
                'min': 'cum_min',
                'max': 'cum_max'
            }
            polars_method = method_map.get(window_method)
            if not polars_method:
                raise TranslationError(f"Unsupported expanding operation: {window_method}")

            # Create pl.col('col').cum_*()
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='df_pl', ctx=ast.Load()),
                    attr='select',
                    ctx=ast.Load()
                ),
                args=[
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='pl', ctx=ast.Load()),
                                    attr='col',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Constant(value=col_name)],
                                keywords=[]
                            ),
                            attr=polars_method,
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    )
                ],
                keywords=[]
            )
        else:
            raise TranslationError(f"Unsupported window operation type: {window_type}")

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

    def _transform_datetime_accessor(self, node: ast.Attribute) -> ast.AST:
        """Transform datetime accessor (dt.xxx) to its polars equivalent."""
        # Get the datetime property being accessed
        dt_attr = node.attr

        # Get the column reference
        if isinstance(node.value.value, ast.Subscript):
            col_name = node.value.value.slice.value
        else:
            col_name = node.value.value.id

        self.needs_polars_import = True

        # Create pl.col('col_name').dt.method() expression
        inner_expr = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='pl', ctx=ast.Load()),
                            attr='col',
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=col_name)],
                        keywords=[]
                    ),
                    attr='dt',
                    ctx=ast.Load()
                ),
                attr=dt_attr,
                ctx=ast.Load()
            ),
            args=[],  # DateTime methods in Polars are called without arguments
            keywords=[]
        )

        # Always wrap in df_pl.select(...)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='df_pl', ctx=ast.Load()),
                attr='select',
                ctx=ast.Load()
            ),
            args=[inner_expr],
            keywords=[]
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Visit an attribute node and transform DataFrame variable names and special attributes."""
        # Handle datetime accessors (df['date'].dt.year)
        if (isinstance(node.value, ast.Attribute) and 
            isinstance(node.value.value, (ast.Subscript, ast.Name)) and 
            node.value.attr == 'dt'):
            return self._transform_datetime_accessor(node)
        
        # Handle DataFrame variable names
        if isinstance(node.value, ast.Name) and node.value.id in self.dataframe_vars:
            return ast.Attribute(
                value=ast.Name(id=f"{node.value.id}_pl", ctx=node.value.ctx),
                attr=node.attr,
                ctx=node.ctx
            )
        return self.generic_visit(node) 