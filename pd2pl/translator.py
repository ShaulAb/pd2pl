"""Core translator module for converting pandas code to polars."""
import ast
from typing import Any, Dict, Optional, Set, Union, List
from enum import Enum

from .errors import UnsupportedPandasUsageError, TranslationError
from .mapping import function_maps, method_maps, FUNCTION_TRANSLATIONS
from .logging import logger


class PandasToPolarsTransformer(ast.NodeTransformer):
    """AST transformer that converts pandas operations to polars operations."""
    
    def __init__(self):
        super().__init__()
        self.dataframe_vars: Set[str] = {'df', 'df_left', 'df_right', 'df_right_diffkey'}
        self.pandas_aliases: Set[str] = {'pd', 'pandas'}
        self.needs_polars_import = False
        self.needs_selector_import = False
        self.in_filter_context = False
        
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit a name node and transform DataFrame variable names."""
        if node.id in self.dataframe_vars:
            transformed_name = f"{node.id}_pl"
            return ast.Name(id=transformed_name, ctx=node.ctx)
        return node
        
    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Visit a function call node and transform pandas functions to polars."""
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            base_name = node.func.value.id
            func_name = node.func.attr

            if base_name in self.pandas_aliases and func_name in FUNCTION_TRANSLATIONS:
                try:
                    translator_func = FUNCTION_TRANSLATIONS[func_name]
                    result_node = translator_func(node, visitor=self) 
                    return result_node
                except TranslationError as e:
                    logger.exception(f"TranslationError during {func_name} translation:")
                    raise e
                except Exception as e:
                    logger.exception(f"Unexpected error during {func_name} translation:")
                    raise TranslationError(f"Internal error translating {func_name}") from e
                
        if self.in_filter_context and isinstance(node.func, ast.Attribute) and node.func.attr == 'isin':
            base = node.func.value
            is_col_selection_base = False
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) and base.value.id in self.dataframe_vars:
                is_col_selection_base = True
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id in self.dataframe_vars:
                is_col_selection_base = True
                 
            if is_col_selection_base:
                transformed_base = self.visit(node.func.value)
                
                self.needs_polars_import = True
                polars_isin_call = ast.Call(
                    func=ast.Attribute(
                        value=transformed_base,
                        attr='is_in',
                        ctx=ast.Load()
                    ),
                    args=node.args,
                    keywords=node.keywords
                )
                return polars_isin_call
        
        elif self.in_filter_context and isinstance(node.func, ast.Attribute) and node.func.attr in ('isna', 'notna'):
            base = node.func.value
            is_col_selection_base = False
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) and base.value.id in self.dataframe_vars:
                 is_col_selection_base = True
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id in self.dataframe_vars:
                 is_col_selection_base = True
                 
            if is_col_selection_base:
                transformed_base = self.visit(node.func.value)
                
                polars_method_name = 'is_null' if node.func.attr == 'isna' else 'is_not_null'
                
                self.needs_polars_import = True
                polars_null_check_call = ast.Call(
                    func=ast.Attribute(
                        value=transformed_base,
                        attr=polars_method_name, 
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
                return polars_null_check_call
        elif self.in_filter_context and \
             isinstance(node.func, ast.Attribute) and \
             isinstance(node.func.value, ast.Attribute) and \
             node.func.value.attr == 'str':
            
            pandas_method_name = node.func.attr
            method_map = {
                'contains': 'contains',
                'startswith': 'starts_with',
                'endswith': 'ends_with'
            }
            polars_method_name = method_map.get(pandas_method_name)

            if polars_method_name:
                base = node.func.value.value 
                is_col_selection_base = False
                if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name) and base.value.id in self.dataframe_vars:
                     is_col_selection_base = True
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id in self.dataframe_vars:
                     is_col_selection_base = True

                if is_col_selection_base:
                    transformed_base = self.visit(base)
                    
                    self.needs_polars_import = True
                    polars_str_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=transformed_base,
                                attr='str',
                                ctx=ast.Load()
                            ),
                            attr=polars_method_name, 
                            ctx=ast.Load()
                        ),
                        args=node.args,  
                        keywords=node.keywords 
                    )
                    return polars_str_call
        elif isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Attribute) and
                node.func.value.attr == 'str'):
                return self._transform_string_method(node)

            if (isinstance(node.func.value, ast.Call) and
                isinstance(node.func.value.func, ast.Attribute) and
                node.func.value.func.attr in {'rolling', 'expanding'}):
                return self._transform_window_operation(node)

            if (isinstance(node.func.value, ast.Call) and
                isinstance(node.func.value.func, ast.Attribute) and
                node.func.value.func.attr == 'groupby'):
                return self._transform_groupby_agg(node)

            if isinstance(node.func.value, ast.Name):
                return self._transform_dataframe_method(node)
            
        return self.generic_visit(node)
        
    def _transform_groupby_agg(self, node: ast.Call) -> ast.AST:
        agg_method = node.func.attr
        groupby_call_node = node.func.value

        transformed_groupby = self.visit_Call(groupby_call_node)

        # Use mapping layer for 'agg' method
        if agg_method == 'agg':
            translation = method_maps.get_method_translation('agg')
            if translation and translation.method_chain:
                args_dict = self._args_to_dict(node)
                chain = translation.method_chain(node.args, args_dict)
                current_node = transformed_groupby
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
            else:
                raise TranslationError("No mapping for groupby.agg")
        # Fallback for simple aggregations (sum, mean, etc.)
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
        final_node = ast.Call(
            func=ast.Attribute(
                value=transformed_groupby,
                attr='agg',
                ctx=ast.Load()
            ),
            args=[agg_exprs[0]],
            keywords=[]
        )
        return final_node
        
    def _transform_dataframe_method(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func.value, ast.Name):
            return self.generic_visit(node)
            
        var_name = node.func.value.id
        method_name = node.func.attr
        
        if hasattr(node.func.value, 'attr') and node.func.value.attr == 'str':
            return self._transform_string_method(node)
            
        if method_maps.is_special_method(method_name):
            raise UnsupportedPandasUsageError(f"Pandas method '{method_name}' is not yet supported")
            
        translation = method_maps.get_method_translation(method_name)
        if not translation:
            raise UnsupportedPandasUsageError(f"Pandas method '{method_name}' has no direct polars equivalent")
            
        self.dataframe_vars.add(var_name)
        
        self.needs_polars_import = True
        if translation.requires_selector:
            self.needs_selector_import = True

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

        new_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=f"{var_name}_pl", ctx=ast.Load()),
                attr=translation.polars_method,
                ctx=ast.Load()
            ),
            args=node.args,
            keywords=node.keywords
        )
            
        if translation.argument_map:
            new_keywords = []
            original_keywords = new_node.keywords
            for kw in original_keywords:
                if kw.arg in translation.argument_map:
                    if translation.argument_map[kw.arg] is not None:
                        new_keywords.append(ast.keyword(
                            arg=translation.argument_map[kw.arg],
                            value=kw.value
                        ))
                else:
                    new_keywords.append(kw)
            new_node.keywords = new_keywords
            
        return new_node
        
    def _transform_string_method(self, node: ast.Call) -> ast.AST:
        method_name = node.func.attr
        translation = method_maps.get_method_translation(method_name, is_string_method=True)
        
        if not translation:
            raise UnsupportedPandasUsageError(f"Pandas string method '{method_name}' is not supported")
            
        if isinstance(node.func.value.value, ast.Subscript):
            col_name = node.func.value.value.slice.value
        else:
            col_name = node.func.value.value.id

        self.needs_polars_import = True

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
                    attr='str',
                    ctx=ast.Load()
                ),
                attr=translation.polars_method,
                ctx=ast.Load()
            ),
            args=node.args,
            keywords=node.keywords
        )

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
        window_method = node.func.attr
        window_call = node.func.value
        window_type = window_call.func.attr
        
        if isinstance(window_call.func.value, ast.Subscript):
            col_name = window_call.func.value.slice.value
        else:
            raise TranslationError("Window operations must be called on a column")

        self.needs_polars_import = True
        
        if window_type == 'rolling':
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
            method_map = {
                'sum': 'cum_sum',
                'mean': 'cum_mean',
                'min': 'cum_min',
                'max': 'cum_max'
            }
            polars_method = method_map.get(window_method)
            if not polars_method:
                raise TranslationError(f"Unsupported expanding operation: {window_method}")

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
        args_dict = {}
        for i, arg in enumerate(node.args):
            args_dict[f"arg{i}"] = arg
        for keyword in node.keywords:
            args_dict[keyword.arg] = keyword.value
        return args_dict
        
    @staticmethod
    def _convert_arg_value(value: Any) -> ast.AST:
        if isinstance(value, ast.AST):
            return value
        elif isinstance(value, (list, tuple)):
            return ast.List(elts=[ast.Constant(value=x) for x in value], ctx=ast.Load())
        else:
            return ast.Constant(value=value) 

    def _transform_datetime_accessor(self, node: ast.Attribute) -> ast.AST:
        dt_attr = node.attr

        if isinstance(node.value.value, ast.Subscript):
            col_name = node.value.value.slice.value
        else:
            col_name = node.value.value.id

        self.needs_polars_import = True

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
            args=[],
            keywords=[]
        )

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
        if (isinstance(node.value, ast.Attribute) and 
            isinstance(node.value.value, (ast.Subscript, ast.Name)) and 
            node.value.attr == 'dt'):
            return self._transform_datetime_accessor(node)
        
        is_df_attribute_access = isinstance(node.value, ast.Name) and node.value.id in self.dataframe_vars

        if is_df_attribute_access and self.in_filter_context:
            self.needs_polars_import = True
            col_name = node.attr
            pl_col_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='pl', ctx=ast.Load()),
                    attr='col',
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=col_name)],
                keywords=[]
            )
            return pl_col_call

        if is_df_attribute_access:
             node.value = self.visit(node.value)
             return node

        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        is_df_base = isinstance(node.value, ast.Name) and node.value.id in self.dataframe_vars

        if is_df_base:
            is_single_col_select = isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
            is_multi_col_select = isinstance(node.slice, (ast.List, ast.Tuple)) and all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in node.slice.elts
            )
            is_boolean_series = isinstance(node.slice, ast.Name) 
            is_filter_slice = not is_single_col_select and not is_multi_col_select or is_boolean_series

            if self.in_filter_context and is_single_col_select:
                self.needs_polars_import = True
                col_name = node.slice.value
                pl_col_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='pl', ctx=ast.Load()),
                        attr='col',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=col_name)],
                    keywords=[]
                )
                return pl_col_call

            elif not self.in_filter_context and (is_single_col_select or is_multi_col_select):
                node.value = self.visit(node.value)
                return node

            elif is_filter_slice:
                self.needs_polars_import = True
                
                _old_context = self.in_filter_context
                self.in_filter_context = True
                
                transformed_condition = self.visit(node.slice)
                
                self.in_filter_context = _old_context
                
                transformed_base = self.visit(node.value)
                
                filter_call = ast.Call(
                    func=ast.Attribute(
                        value=transformed_base,
                        attr='filter',
                        ctx=ast.Load()
                    ),
                    args=[transformed_condition],
                    keywords=[]
                )
                return filter_call
            
            elif is_boolean_series and not self.in_filter_context:
                 self.needs_polars_import = True
                 _old_context = self.in_filter_context
                 self.in_filter_context = True
                 transformed_condition = self.visit(node.slice)
                 self.in_filter_context = _old_context
                 transformed_base = self.visit(node.value)
                 
                 filter_call = ast.Call(
                    func=ast.Attribute(
                        value=transformed_base,
                        attr='filter',
                        ctx=ast.Load()
                    ),
                    args=[transformed_condition],
                    keywords=[]
                 )
                 return filter_call

        return self.generic_visit(node) 
    