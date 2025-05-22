"""Core translator module for converting pandas code to polars."""
import ast
from typing import Any, Dict, Optional, Set, List
import importlib.util
import logging
import warnings

from .errors import UnsupportedPandasUsageError, TranslationError
from .mapping import function_maps, method_maps, FUNCTION_TRANSLATIONS
from .logging import logger
from .mapping.dtype_maps import to_polars_dtype, CONSTRUCTOR_MAP
from .schema_tracking import SchemaState, SchemaRegistry
from .chain_tracking import ChainRegistry, ChainNode
from .chain_preprocessing import preprocess_chains

try:
    import astroid
    ASTROID_AVAILABLE = True
except ImportError:
    ASTROID_AVAILABLE = False


class PandasToPolarsTransformer(ast.NodeTransformer):
    """AST transformer that converts pandas operations to polars operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.dataframe_vars: Set[str] = {'df', 'df_left', 'df_right', 'df_right_diffkey'}
        self.pandas_aliases: Set[str] = {'pd', 'pandas'}
        self.numpy_aliases: Set[str] = {'np'}
        self.needs_polars_import = False
        self.needs_selector_import = False
        self.in_filter_context = False
        self.df_count = 0
        # Schema registry for tracking DataFrame schemas
        self.schema_registry = SchemaRegistry()
        # Chain registry for tracking method chains
        self.chain_registry: Optional[ChainRegistry] = None
        # Skip nodes that have been processed as part of chains
        self.processed_chain_nodes: Set[int] = set()
        
    def preprocess_tree(self, tree: ast.AST) -> ast.AST:
        """Preprocess the AST to identify and analyze method chains."""
        logger.debug("Starting chain preprocessing phase")
        chain_registry, schema_registry = preprocess_chains(tree, self.dataframe_vars)
        self.chain_registry = chain_registry
        self.schema_registry = schema_registry
        logger.debug("Preprocessing complete. Ready for transformation.")
        return tree
        
    def process(self, tree: ast.AST) -> ast.AST:
        """Process the entire AST with preprocessing and transformation."""
        # Phase 1: Preprocess to identify chains
        preprocessed_tree = self.preprocess_tree(tree)
        
        # Before transforming, check if we have a top-level chain with groupby + agg + sort
        if self.chain_registry:
            for chain_id, nodes in self.chain_registry.chains_by_id.items():
                sorted_nodes = sorted(nodes, key=lambda n: n.position)
                methods = [n.method_name for n in sorted_nodes]
                
                # If this is a groupby+mean+sort_values chain, handle it directly
                if 'groupby' in methods and 'mean' in methods and 'sort_values' in methods:
                    logger.debug(f"Using specialized transformation for entire groupby+agg+sort chain: {' -> '.join(methods)}")
                    # Find the root expression
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Expr) and hasattr(node, 'value'):
                            if id(node.value) == id(sorted_nodes[-1].node):  # Check if this is our chain's end node
                                # Replace with our custom chain transformation
                                root_node = sorted_nodes[0]
                                logger.debug(f"Transforming entire chain starting at {root_node.method_name}")
                                
                                # Use our specialized transformation with schema context
                                result = self._transform_complete_groupby_chain(sorted_nodes)
                                
                                # Replace the node with our result
                                transformed_tree = tree
                                for body_item in transformed_tree.body:
                                    if isinstance(body_item, ast.Expr) and id(body_item.value) == id(node.value):
                                        body_item.value = result
                                return transformed_tree
        
        # Default: Transform the AST
        transformed_tree = self.visit(preprocessed_tree)
        
        return transformed_tree
        
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit a name node and transform DataFrame/Series variable names if configured."""
        if node.id in self.dataframe_vars:
            if self.config.get('rename_dataframe', False):
                transformed_name = f"{node.id}_pl"
                return ast.Name(id=transformed_name, ctx=node.ctx)
        return node
        
    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Skip nodes that have already been processed as part of a chain
        if hasattr(self, 'processed_chain_nodes') and id(node) in self.processed_chain_nodes:
            logger.debug(f"Skipping already processed chain node: {id(node)}")
            return node
            
        # Check if this node is part of a method chain
        if self.chain_registry and self.chain_registry.is_in_chain(node):
            chain_node = self.chain_registry.get_node(node)
            chain = self.chain_registry.get_chain_for_node(node)
            
            # Only process from the chain root to avoid duplicate processing
            if chain_node and chain and chain_node.position == 0:
                logger.debug(f"Processing chain from root: {chain_node.method_name}")
                
                # Process the entire chain at once
                sorted_chain = sorted(chain, key=lambda n: n.position)
                
                # Check if the chain contains a groupby method
                methods = [n.method_name for n in sorted_chain]
                if 'groupby' in methods:
                    logger.debug(f"Using specialized groupby chain transformation for {' -> '.join(methods)}")
                    transformed_node = self._transform_groupby_chain(sorted_chain)
                else:
                    # Use a generic chain transformation for other cases
                    transformed_node = self._transform_generic_chain(sorted_chain)
                
                # Mark all chain nodes as processed to avoid duplicate processing
                self.processed_chain_nodes = self.processed_chain_nodes.union({id(n.node) for n in chain})
                
                return transformed_node
            elif chain_node and chain:
                # Skip non-root nodes, they will be processed when we start from the root
                logger.debug(f"Skipping non-root chain node: {chain_node.method_name}")
                return node
                
        # If already a polars call, just visit args/keywords and return
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pl':
            new_args = [self.visit(arg) for arg in node.args]
            new_keywords = [ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
            new_call = ast.Call(
                func=node.func,
                args=new_args,
                keywords=new_keywords
            )
            ast.copy_location(new_call, node)
            return new_call
        # If numpy call, just visit args/keywords and return
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
            new_args = [self.visit(arg) for arg in node.args]
            new_keywords = [ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
            new_call = ast.Call(
                func=node.func,
                args=new_args,
                keywords=new_keywords
            )
            ast.copy_location(new_call, node)
            return new_call
        func_repr = None
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                func_repr = f"{node.func.value.id}.{node.func.attr}"
            else:
                func_repr = f"<complex>.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            func_repr = node.func.id
        logger.debug(f"visit_Call: visiting {func_repr}")
        # Handle known constructor calls (e.g., pd.Categorical([...]))
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            mod = node.func.value.id
            attr = node.func.attr
            logger.debug(f"visit_Call: checking CONSTRUCTOR_MAP for ({mod}, {attr})")
            if (mod, attr) in CONSTRUCTOR_MAP:
                mapping = CONSTRUCTOR_MAP[(mod, attr)]
                logger.debug(f"visit_Call: matched CONSTRUCTOR_MAP for ({mod}, {attr}) -> {mapping}")
                if callable(mapping):
                    return mapping(node, context=self)
                else:
                    target_mod, target_attr = mapping
                    new_func = ast.Attribute(
                        value=ast.Name(id=target_mod, ctx=ast.Load()),
                        attr=target_attr,
                        ctx=ast.Load()
                    )
                    new_call = ast.Call(
                        func=new_func,
                        args=[self.visit(arg) for arg in node.args],
                        keywords=[ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
                    )
                    ast.copy_location(new_call, node)
                    return new_call
        # Handle pd.Series(..., dtype='category') and pd.DataFrame(..., dtype='category')
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in self.pandas_aliases and node.func.attr in {'Series', 'DataFrame'}:
                # Look for dtype keyword
                for kw in node.keywords:
                    if kw.arg == 'dtype':
                        polars_dtype = to_polars_dtype(kw.value)
                        if polars_dtype is not None:
                            self.needs_polars_import = True
                            # Replace pd.Series/DataFrame with pl.Series/DataFrame and dtype=pl.Categorical
                            new_func = ast.Attribute(
                                value=ast.Name(id='pl', ctx=ast.Load()),
                                attr=node.func.attr,
                                ctx=ast.Load()
                            )
                            # RECURSIVELY VISIT ALL ARGS AND KEYWORDS
                            new_args = [self.visit(arg) for arg in node.args]
                            new_keywords = [
                                ast.keyword(arg='dtype', value=polars_dtype) if k.arg == 'dtype' else ast.keyword(arg=k.arg, value=self.visit(k.value))
                                for k in node.keywords
                            ]
                            new_call = ast.Call(
                                func=new_func,
                                args=new_args,
                                keywords=new_keywords
                            )
                            ast.copy_location(new_call, node)
                            return new_call
        # Handle astype('category') and similar dtype conversions
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'astype':
            # e.g., df['col'].astype('category')
            if node.args:
                polars_dtype = to_polars_dtype(node.args[0])
                if polars_dtype is not None:
                    self.needs_polars_import = True
                    # Replace astype('category') with cast(pl.Categorical)
                    new_call = ast.Call(
                        func=ast.Attribute(
                            value=self.visit(node.func.value),
                            attr='cast',
                            ctx=ast.Load()
                        ),
                        args=[polars_dtype],
                        keywords=[]
                    )
                    ast.copy_location(new_call, node)
                    return new_call
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
        
    def _transform_method_chain(self, chain: List[ChainNode]) -> ast.AST:
        """Transform an entire method chain at once."""
        if not chain:
            return None
            
        # Sort nodes by position
        sorted_nodes = sorted(chain, key=lambda n: n.position)
        logger.debug(f"Transforming method chain: {' -> '.join(n.method_name for n in sorted_nodes)}")
        
        # Handle different types of chains based on their methods
        methods = [node.method_name for node in sorted_nodes]
        
        # Check for groupby-based chains
        if 'groupby' in methods:
            return self._transform_groupby_chain(sorted_nodes)
            
        # Generic chain handling for other cases
        return self._transform_generic_chain(sorted_nodes)
    
    def _transform_complete_groupby_chain(self, chain_nodes: List[ChainNode]) -> ast.AST:
        """Transform a complete chain of groupby+aggregation+sort with full schema context."""
        logger.debug(f"Transforming complete groupby chain with {len(chain_nodes)} methods")
        
        # Identify the main components of the chain
        groupby_node = None
        agg_node = None
        sort_node = None
        
        for node in chain_nodes:
            if node.method_name == 'groupby':
                groupby_node = node
            elif node.method_name in ['mean', 'sum', 'min', 'max', 'count']:
                agg_node = node
            elif node.method_name == 'sort_values':
                sort_node = node
                
        if not (groupby_node and agg_node):
            logger.warning("Expected groupby and aggregation nodes not found in chain")
            return self.generic_visit(chain_nodes[0].node)
        
        # Extract DataFrame and column information
        root_node = chain_nodes[0]
        df_var = None
        selected_columns = []
        
        # Get DataFrame and selected columns
        if isinstance(root_node.node.func.value, ast.Subscript) and isinstance(root_node.node.func.value.value, ast.Name):
            df_var = root_node.node.func.value.value.id
            selected_columns = self._extract_columns_from_selection(root_node.node.func.value)
        elif isinstance(root_node.node.func.value, ast.Name):
            df_var = root_node.node.func.value.id
            
        logger.debug(f"Chain base: {df_var} with columns {selected_columns}")
        
        # Start building the transformed chain
        current_node = None
        if selected_columns:
            current_node = ast.Subscript(
                value=ast.Name(id=df_var, ctx=ast.Load()),
                slice=ast.List(
                    elts=[ast.Constant(value=col) for col in selected_columns],
                    ctx=ast.Load()
                ),
                ctx=ast.Load()
            )
        else:
            current_node = ast.Name(id=df_var, ctx=ast.Load())
            
        # Add group_by operation
        group_keys = self._extract_groupby_keys(groupby_node.node)
        current_node = ast.Call(
            func=ast.Attribute(
                value=current_node,
                attr='group_by',  # Transform 'groupby' to 'group_by'
                ctx=ast.Load()
            ),
            args=[ast.Constant(value=key) if isinstance(key, str) else key for key in group_keys],
            keywords=[]
        )
        
        # Add aggregation
        agg_method = agg_node.method_name
        current_node = ast.Call(
            func=ast.Attribute(
                value=current_node,
                attr=agg_method,
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        
        # Add sort if present
        if sort_node:
            # Get schema after aggregation
            schema = agg_node.schema_after
            
            # Extract sort parameters
            args_dict = self._args_to_dict(sort_node.node)
            sort_args = []
            sort_kwargs = {}
            
            # Handle 'by' parameter with proper column renaming
            if 'by' in args_dict:
                by_arg = args_dict['by']
                if isinstance(by_arg, ast.Constant) and isinstance(by_arg.value, str):
                    col_name = by_arg.value
                    # Get the renamed column from schema
                    if schema and col_name in schema.aggregated_columns:
                        renamed_col = schema.aggregated_columns[col_name]
                        logger.debug(f"Sort: Renaming column {col_name} to {renamed_col}")
                        sort_args.append(ast.Constant(value=renamed_col))
                    else:
                        # Keep original name if no renaming found
                        sort_args.append(ast.Constant(value=col_name))
                elif isinstance(by_arg, ast.List):
                    # List of columns
                    col_names = [elt.value for elt in by_arg.elts if isinstance(elt, ast.Constant)]
                    renamed_cols = []
                    for col in col_names:
                        if schema and col in schema.aggregated_columns:
                            renamed = schema.aggregated_columns[col]
                            logger.debug(f"Sort: Renaming column {col} to {renamed}")
                            renamed_cols.append(renamed)
                        else:
                            renamed_cols.append(col)
                    sort_args.append(
                        ast.List(
                            elts=[ast.Constant(value=col) for col in renamed_cols],
                            ctx=ast.Load()
                        )
                    )
            
            # Handle 'ascending' parameter - change to 'descending' with inverted value
            if 'ascending' in args_dict:
                ascending = args_dict['ascending']
                if isinstance(ascending, ast.Constant) and isinstance(ascending.value, bool):
                    sort_kwargs['descending'] = ast.Constant(value=not ascending.value)
            
            # Add sort call with renamed columns
            current_node = ast.Call(
                func=ast.Attribute(
                    value=current_node,
                    attr='sort',  # Transform 'sort_values' to 'sort'
                    ctx=ast.Load()
                ),
                args=sort_args,
                keywords=[ast.keyword(arg=k, value=v) for k, v in sort_kwargs.items()]
            )
        
        # Return the complete transformed chain
        return current_node
    
    def _transform_groupby_chain(self, chain_nodes: List[ChainNode]) -> ast.AST:
        """Transform a chain that includes groupby operations."""
        logger.debug(f"Transforming groupby chain with {len(chain_nodes)} methods")
        
        # Identify the main components of the chain
        groupby_node = None
        agg_node = None
        sort_node = None
        
        for node in chain_nodes:
            if node.method_name == 'groupby':
                groupby_node = node
            elif node.method_name in ['mean', 'sum', 'min', 'max', 'count']:
                agg_node = node
            elif node.method_name == 'sort_values':
                sort_node = node
        
        if not groupby_node:
            logger.warning("Expected groupby node not found in chain")
            return self.generic_visit(chain_nodes[0].node)
            
        # Get initial part of the chain before groupby
        root_node = chain_nodes[0]
        df_var = None
        selected_columns = []
        
        # Extract DataFrame name and selected columns
        if isinstance(root_node.node.func.value, ast.Name):
            df_var = root_node.node.func.value.id
        elif (isinstance(root_node.node.func.value, ast.Subscript) and 
              isinstance(root_node.node.func.value.value, ast.Name)):
            df_var = root_node.node.func.value.value.id
            
            # Extract selected columns
            slice_node = root_node.node.func.value.slice
            if isinstance(slice_node, ast.Constant):
                selected_columns = [slice_node.value]
            elif isinstance(slice_node, ast.List):
                selected_columns = [
                    elt.value for elt in slice_node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                
        logger.debug(f"Chain base: {df_var} with columns {selected_columns}")
        
        # Start building the chain from the DataFrame
        if selected_columns:
            # Handle column selection
            base_node = ast.Subscript(
                value=ast.Name(id=df_var, ctx=ast.Load()),
                slice=ast.List(
                    elts=[ast.Constant(value=col) for col in selected_columns],
                    ctx=ast.Load()
                ),
                ctx=ast.Load()
            )
        else:
            # Direct DataFrame reference
            base_node = ast.Name(id=df_var, ctx=ast.Load())
            
        # Transform the groupby operation
        group_keys = self._extract_groupby_keys(groupby_node.node)
        logger.debug(f"Groupby keys: {group_keys}")
        
        # Start building the transformed chain
        current_node = base_node
        
        # Add group_by operation
        current_node = ast.Call(
            func=ast.Attribute(
                value=current_node,
                attr='group_by',  # Transform 'groupby' to 'group_by'
                ctx=ast.Load()
            ),
            args=[ast.Constant(value=key) if isinstance(key, str) else key for key in group_keys],
            keywords=[]
        )
        
        # Add aggregation if present
        if agg_node:
            agg_method = agg_node.method_name
            
            # Get schema after groupby to determine column references
            schema = groupby_node.schema_after
            
            # For each non-group column, create an aggregation expression
            agg_exprs = []
            for col in schema.columns:
                if col not in schema.group_keys:
                    logger.debug(f"Adding aggregation for column: {col}")
                    agg_exprs.append(
                        ast.Call(
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
                                attr=agg_method,
                                ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[]
                        )
                    )
            
            # If no specific columns, use pl.all()
            if not agg_exprs:
                logger.debug("Using pl.all() for aggregation")
                agg_exprs = [
                    ast.Call(
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
                    )
                ]
            
            # Add agg() call
            current_node = ast.Call(
                func=ast.Attribute(
                    value=current_node,
                    attr='agg',
                    ctx=ast.Load()
                ),
                args=agg_exprs if len(agg_exprs) > 1 else [agg_exprs[0]],
                keywords=[]
            )
        
        # Add sort if present
        if sort_node:
            # Get schema after aggregation to resolve column names
            schema = agg_node.schema_after if agg_node else groupby_node.schema_after
            
            logger.debug(f"Resolving sort columns using schema. Columns: {schema.columns}")
            logger.debug(f"Aggregated column mappings: {schema.aggregated_columns}")
            
            # Extract sort parameters
            args_dict = self._args_to_dict(sort_node.node)
            sort_args = []
            sort_kwargs = {}
            
            # Handle 'by' parameter with schema resolution
            if 'by' in args_dict:
                by_arg = args_dict['by']
                if isinstance(by_arg, ast.Constant) and isinstance(by_arg.value, str):
                    # Single column
                    col_name = by_arg.value
                    # Check for renamed columns after aggregation
                    if col_name in schema.aggregated_columns:
                        resolved_col = schema.aggregated_columns[col_name]
                        logger.debug(f"Found direct mapping for sort column: {col_name} -> {resolved_col}")
                    else:
                        resolved_col = schema.resolve_column_reference(col_name)
                        logger.debug(f"Resolved sort column via schema: {col_name} -> {resolved_col}")
                    
                    logger.debug(f"Final resolved sort column: {col_name} -> {resolved_col}")
                    sort_args.append(ast.Constant(value=resolved_col))
                elif isinstance(by_arg, ast.List):
                    # List of columns
                    col_names = [
                        elt.value for elt in by_arg.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    
                    # Resolve each column with direct lookup in aggregated_columns first
                    resolved_cols = []
                    for col in col_names:
                        if col in schema.aggregated_columns:
                            resolved = schema.aggregated_columns[col]
                            logger.debug(f"Found direct mapping for sort column: {col} -> {resolved}")
                            resolved_cols.append(resolved)
                        else:
                            resolved = schema.resolve_column_reference(col)
                            logger.debug(f"Resolved sort column via schema: {col} -> {resolved}")
                            resolved_cols.append(resolved)
                    
                    logger.debug(f"Final resolved sort columns: {col_names} -> {resolved_cols}")
                    sort_args.append(
                        ast.List(
                            elts=[ast.Constant(value=col) for col in resolved_cols],
                            ctx=ast.Load()
                        )
                    )
            
            # Handle 'ascending' parameter
            if 'ascending' in args_dict:
                ascending = args_dict['ascending']
                if isinstance(ascending, ast.Constant) and isinstance(ascending.value, bool):
                    # Convert to 'descending' with inverted value
                    sort_kwargs['descending'] = ast.Constant(value=not ascending.value)
            
            # Add sort call
            current_node = ast.Call(
                func=ast.Attribute(
                    value=current_node,
                    attr='sort',  # Transform 'sort_values' to 'sort'
                    ctx=ast.Load()
                ),
                args=sort_args,
                keywords=[
                    ast.keyword(arg=k, value=v) for k, v in sort_kwargs.items()
                ]
            )
        
        # Return the complete transformed chain
        return current_node
    
    def _transform_generic_chain(self, chain_nodes: List[ChainNode]) -> ast.AST:
        """Transform a generic method chain."""
        logger.debug(f"Transforming generic chain with methods: {[n.method_name for n in chain_nodes]}")
        
        # Start with the base node
        root_node = chain_nodes[0]
        base_node = None
        
        # Determine the base node type
        if isinstance(root_node.node.func.value, ast.Name):
            # Direct DataFrame method: df.method()
            df_var = root_node.node.func.value.id
            base_node = ast.Name(id=df_var, ctx=ast.Load())
        elif (isinstance(root_node.node.func.value, ast.Subscript) and 
              isinstance(root_node.node.func.value.value, ast.Name)):
            # Column selection: df['col'].method() or df[['col1', 'col2']].method()
            df_var = root_node.node.func.value.value.id
            base_node = self.visit(root_node.node.func.value)
        else:
            # Complex base, use generic visit
            logger.warning(f"Complex chain base, using generic visit for {root_node.method_name}")
            return self.generic_visit(root_node.node)
            
        # Build the chain by applying methods sequentially
        current_node = base_node
        
        for chain_node in chain_nodes:
            # Get method translation
            method_name = chain_node.method_name
            translation = method_maps.get_method_translation(method_name)
            
            if not translation:
                logger.warning(f"No translation for method {method_name}, using original")
                polars_method = method_name
            else:
                polars_method = translation.polars_method
                
            logger.debug(f"Translating method {method_name} -> {polars_method}")
            
            # Extract args and kwargs
            args = [self.visit(arg) for arg in chain_node.node.args]
            kwargs = [
                ast.keyword(arg=kw.arg, value=self.visit(kw.value))
                for kw in chain_node.node.keywords
                # Skip 'inplace' parameter
                if kw.arg != 'inplace'
            ]
            
            # Apply argument mappings if available
            if translation and translation.argument_map:
                mapped_kwargs = []
                for kw in kwargs:
                    if kw.arg in translation.argument_map:
                        mapped_name = translation.argument_map[kw.arg]
                        if mapped_name is not None:
                            mapped_kwargs.append(ast.keyword(arg=mapped_name, value=kw.value))
                    else:
                        mapped_kwargs.append(kw)
                kwargs = mapped_kwargs
            
            # Apply the method
            current_node = ast.Call(
                func=ast.Attribute(
                    value=current_node,
                    attr=polars_method,
                    ctx=ast.Load()
                ),
                args=args,
                keywords=kwargs
            )
        
        return current_node

    def _transform_groupby_agg(self, node: ast.Call) -> ast.AST:
        agg_method = node.func.attr
        groupby_call_node = node.func.value

        # Extract group keys
        group_keys = self._extract_groupby_keys(groupby_call_node)
        logger.debug(f"Extracted group keys: {group_keys}")
        
        # Get DataFrame variable name and extract columns
        df_var = None
        selected_columns = []
        
        if isinstance(groupby_call_node, ast.Call) and isinstance(groupby_call_node.func, ast.Attribute):
            if isinstance(groupby_call_node.func.value, ast.Name):
                # Simple case: df.groupby()
                df_var = groupby_call_node.func.value.id
            elif isinstance(groupby_call_node.func.value, ast.Subscript):
                # Case: df[['col1', 'col2']].groupby()
                if isinstance(groupby_call_node.func.value.value, ast.Name):
                    df_var = groupby_call_node.func.value.value.id
                    selected_columns = self._extract_columns_from_selection(groupby_call_node.func.value)
                    logger.debug(f"Extracted selected columns from {df_var}: {selected_columns}")
        
        logger.debug(f"DataFrame variable: {df_var}")
        
        # Get or create schema
        schema = None
        if df_var:
            schema = self.schema_registry.get_schema(df_var)
            if not schema:
                schema = self.schema_registry.register_dataframe(df_var)
                logger.debug(f"Created new schema for {df_var}")
            else:
                logger.debug(f"Using existing schema for {df_var}: columns={schema.columns}")
            
            # Create a copy for this chain
            schema = schema.copy()
            
            # Apply selection if columns were specified
            if selected_columns:
                schema.apply_selection(selected_columns)
                logger.debug(f"Applied selection to schema: {selected_columns}")
            
            # Apply groupby and aggregation
            schema.apply_groupby(group_keys)
            logger.debug(f"Applied groupby to schema with keys: {group_keys}")
            
            schema.apply_aggregation(agg_method)
            logger.debug(f"Applied aggregation {agg_method} to schema")
            logger.debug(f"Updated schema columns: {schema.columns}")
            logger.debug(f"Updated schema aggregated columns: {schema.aggregated_columns}")
        
        # Visit the groupby call to transform it
        transformed_groupby = self.visit_Call(groupby_call_node)

        # Use mapping layer for 'agg' method
        if agg_method == 'agg':
            translation = method_maps.get_method_translation('agg')
            if translation and translation.method_chain:
                args_dict = self._args_to_dict(node)
                chain = translation.method_chain(node.args, args_dict)
                
                # Process aggregation arguments to update schema
                if schema and node.args and isinstance(node.args[0], ast.Dict):
                    agg_dict = {}
                    for col_node, agg_node in zip(node.args[0].keys, node.args[0].values):
                        if isinstance(col_node, ast.Constant):
                            col = col_node.value
                            # Single aggregation as string
                            if isinstance(agg_node, ast.Constant):
                                agg_dict[col] = agg_node.value
                            # Multiple aggregations as list
                            elif isinstance(agg_node, ast.List):
                                agg_dict[col] = [
                                    elt.value for elt in agg_node.elts 
                                    if isinstance(elt, ast.Constant)
                                ]
                    
                    schema.apply_aggregation_dict(agg_dict)
                    logger.debug(f"Applied aggregation dict to schema: {agg_dict}")
                
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
                
                # Save schema for the result DataFrame
                result_var = self._get_target_name(node)
                if result_var and schema:
                    self.schema_registry.update_schema(result_var, schema)
                    logger.debug(f"Updated schema for {result_var}")
                    
                return current_node
            else:
                raise TranslationError("No mapping for groupby.agg")
                
        # Fallback for simple aggregations (sum, mean, etc.)
        self.needs_polars_import = True
        
        # Generate specific column expressions for non-group columns
        agg_exprs = []
        if schema:
            # Use schema to generate specific column expressions
            for col in schema.columns:
                if col in schema.group_keys:
                    continue
                    
                # Find the original column name (before aggregation)
                original_col = None
                for agg_col, info in schema.aggregated_columns.items():
                    if agg_col == col:
                        original_col = info['source_col']
                        break
                
                if original_col:
                    logger.debug(f"Adding specific column expression for {original_col} -> {col}")
                    agg_exprs.append(ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='pl', ctx=ast.Load()),
                                    attr='col',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Constant(value=original_col)],
                                keywords=[]
                            ),
                            attr=agg_method,
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    ))
        
        # If no specific columns found, use pl.all()
        if not agg_exprs:
            logger.debug("No specific columns found, using pl.all()")
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
            args=agg_exprs if len(agg_exprs) > 1 else [agg_exprs[0]],
            keywords=[]
        )
        
        # Save schema for the result DataFrame
        result_var = self._get_target_name(node)
        if result_var and schema:
            self.schema_registry.update_schema(result_var, schema)
            logger.debug(f"Updated schema for {result_var}")
            
        return final_node
        
    def _transform_dataframe_method(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func.value, ast.Name):
            return self.generic_visit(node)
            
        var_name = node.func.value.id
        method_name = node.func.attr
        logger.debug(f"_transform_dataframe_method: processing {var_name}.{method_name}")
        
        if hasattr(node.func.value, 'attr') and node.func.value.attr == 'str':
            return self._transform_string_method(node)
            
        if method_maps.is_special_method(method_name):
            raise UnsupportedPandasUsageError(f"Pandas method '{method_name}' is not yet supported")
            
        translation = method_maps.get_method_translation(method_name)
        if not translation:
            raise UnsupportedPandasUsageError(f"Pandas method '{method_name}' has no direct polars equivalent")
            
        logger.debug(f"_transform_dataframe_method: translating {method_name} -> {translation.polars_method}")
        
        self.dataframe_vars.add(var_name)
        
        # Get schema for this DataFrame
        schema = self.schema_registry.get_schema(var_name)
        if not schema:
            # Register with empty schema if not already tracked
            schema = self.schema_registry.register_dataframe(var_name)
        
        self.needs_polars_import = True
        if translation.requires_selector:
            self.needs_selector_import = True

        # Check for inplace parameter
        inplace = False
        args_dict = self._args_to_dict(node)
        if 'inplace' in args_dict:
            inplace_node = args_dict['inplace']
            if isinstance(inplace_node, ast.Constant) and inplace_node.value is True:
                inplace = True
                # Remove inplace from args_dict to prevent it being passed to the polars method
                del args_dict['inplace']
                # Filter out inplace from keywords
                node.keywords = [kw for kw in node.keywords if kw.arg != 'inplace']

        # Create a new schema for this operation chain
        new_schema = schema.copy()
                
        # Update schema for method-specific operations
        if method_name == 'groupby':
            group_keys = self._extract_groupby_keys(node)
            new_schema.apply_groupby(group_keys)
            logger.debug(f"Applied groupby to schema with keys: {group_keys}")
        elif method_name in ['mean', 'sum', 'min', 'max', 'count']:
            new_schema.apply_aggregation(method_name)
            logger.debug(f"Applied aggregation {method_name} to schema")

        if translation.method_chain:
            # Pass schema to the method chain function if needed
            if method_name == 'sort_values':
                logger.debug(f"_transform_dataframe_method: passing schema to sort_values chain")
                chain = translation.method_chain(node.args, args_dict, new_schema)
            else:
                chain = translation.method_chain(node.args, args_dict)
                
            logger.debug(f"_transform_dataframe_method: method chain = {chain}")
                
            if chain:
                current_node = ast.Name(id=f"{var_name}_pl", ctx=ast.Load()) if self.config.get('rename_dataframe', False) else ast.Name(id=var_name, ctx=ast.Load())
                
                # Process each method in the chain
                for i, (method, args, kwargs) in enumerate(chain):
                    logger.debug(f"_transform_dataframe_method: building chain step {i+1}: {method}")
                    
                    # Create the method call
                    current_node = ast.Call(
                        func=ast.Attribute(
                            value=current_node,
                            attr=method,  # This should be the polars method name
                            ctx=ast.Load()
                        ),
                        args=args,
                        keywords=[ast.keyword(arg=k, value=self._convert_arg_value(v)) 
                                for k, v in kwargs.items() if v is not None]
                    )
                
                # Save the schema for the result
                result_var = self._get_target_name(node)
                if result_var:
                    self.schema_registry.update_schema(result_var, new_schema)
                    logger.debug(f"Updated schema for {result_var}")
                elif inplace:
                    self.schema_registry.update_schema(var_name, new_schema)
                    logger.debug(f"Updated schema for {var_name} (inplace)")
                
                # If inplace=True, wrap in assignment
                if inplace:
                    target_name = self._get_target_name(node.func.value)
                    target_node = ast.Name(id=target_name, ctx=ast.Store())
                    ast.copy_location(target_node, node)  # Copy location to the target name
                    result_node = ast.Assign(targets=[target_node], value=current_node)
                    ast.copy_location(result_node, node)  # Copy location to the assignment
                    return result_node
                
                logger.debug(f"_transform_dataframe_method: final node = {ast.dump(current_node)}")
                return current_node

        # For simple method calls (not chains)
        logger.debug(f"_transform_dataframe_method: creating simple method call {translation.polars_method}")
        new_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=f"{var_name}_pl", ctx=ast.Load()) if self.config.get('rename_dataframe', False) else ast.Name(id=var_name, ctx=ast.Load()),
                attr=translation.polars_method,  # This should be the polars method name
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
                        new_kw = ast.keyword(arg=translation.argument_map[kw.arg], value=kw.value)
                        new_keywords.append(new_kw)
                else:
                    new_keywords.append(kw)
            new_node.keywords = new_keywords
            
        # Save the schema
        result_var = self._get_target_name(node)
        if result_var:
            self.schema_registry.update_schema(result_var, new_schema)
            logger.debug(f"Updated schema for {result_var}")
        elif inplace:
            self.schema_registry.update_schema(var_name, new_schema)
            logger.debug(f"Updated schema for {var_name} (inplace)")

        # If inplace=True, wrap in assignment
        if inplace:
            target_name = self._get_target_name(node.func.value)
            target_node = ast.Name(id=target_name, ctx=ast.Store())
            ast.copy_location(target_node, node)  # Copy location to the target name
            result_node = ast.Assign(targets=[target_node], value=new_node)
            ast.copy_location(result_node, node)  # Copy location to the assignment
            return result_node
        
        logger.debug(f"_transform_dataframe_method: final simple node = {ast.dump(new_node)}")
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

        # Apply method_chain transformations if available
        args_list = node.args
        kwargs_dict = {kw.arg: kw.value for kw in node.keywords}
        
        if translation.method_chain:
            # Use the method_chain function to transform arguments
            steps = translation.method_chain(args_list, kwargs_dict)
            if steps and len(steps) > 0:
                # We expect a single step for string methods, with transformed args and kwargs
                method, transformed_args, transformed_kwargs = steps[0]
                # Create the inner expression with transformed args and kwargs
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
                    args=transformed_args,
                    keywords=[ast.keyword(arg=k, value=v) for k, v in transformed_kwargs.items()]
                )
            else:
                # Fallback if method_chain returns empty
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
        else:
            # No method_chain, use original args and keywords
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
                value=ast.Name(id='df_pl', ctx=ast.Load()) if self.config.get('rename_dataframe', False) else ast.Name(id='df', ctx=ast.Load()),
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
                    value=ast.Name(id='df_pl', ctx=ast.Load()) if self.config.get('rename_dataframe', False) else ast.Name(id='df', ctx=ast.Load()),
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
                    value=ast.Name(id='df_pl', ctx=ast.Load()) if self.config.get('rename_dataframe', False) else ast.Name(id='df', ctx=ast.Load()),
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
                value=ast.Name(id='df_pl', ctx=ast.Load()) if self.config.get('rename_dataframe', False) else ast.Name(id='df', ctx=ast.Load()),
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
    
    def is_polars_dataframe_creator_call(self, node):
        """
        Returns True if node is a call to pl.read_csv or pl.DataFrame
        """
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'pl'
            and node.func.attr in {'read_csv', 'DataFrame'}
        )

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        rhs = node.value
        new_df_vars = set()
        # Hybrid logic: AST for simple, astroid for complex
        # 1. Direct pandas DataFrame-creating function call
        if is_pandas_dataframe_creator_call(rhs, self.pandas_aliases):
            # Rewrite to polars
            rewritten_rhs = rewrite_chain_base_to_polars(rhs, self.pandas_aliases)
            if rewritten_rhs is not None:
                # If we rewrote to pl.DataFrame/pl.Series, set needs_polars_import
                self.needs_polars_import = True  # Ensure AUTO strategy adds import
            rhs = rewritten_rhs if rewritten_rhs is not None else rhs
            rhs = self.visit(rhs)  # Ensure recursive visiting of new rhs
            
            # Extract columns if possible
            columns = self._extract_columns_from_df_creation(rhs)
            
            for target in node.targets:
                if isinstance(target, ast.Name):
                    new_df_vars.add(target.id)
                    # Register DataFrame with columns
                    if columns:
                        self.schema_registry.register_dataframe(target.id, columns)
                    
            new_targets = [
                ast.Name(id=f"{t.id}_pl", ctx=ast.Store()) if isinstance(t, ast.Name) and self.config.get('rename_dataframe', False) else t
                for t in node.targets
            ]
            for t, orig in zip(new_targets, node.targets):
                ast.copy_location(t, orig)
            ast.copy_location(rhs, rhs)
            new_assign = ast.Assign(targets=new_targets, value=rhs)
            ast.copy_location(new_assign, node)
            self.dataframe_vars.update(new_df_vars)
            return new_assign
        # 2. Simple method call on known DataFrame variable
        elif is_method_call_on_known_df(rhs, self.dataframe_vars):
            rhs = self.visit(rhs)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    new_df_vars.add(target.id)
            new_targets = [
                ast.Name(id=f"{t.id}_pl", ctx=ast.Store()) if isinstance(t, ast.Name) and self.config.get('rename_dataframe', False) else t
                for t in node.targets
            ]
            for t, orig in zip(new_targets, node.targets):
                ast.copy_location(t, orig)
            ast.copy_location(rhs, rhs)
            new_assign = ast.Assign(targets=new_targets, value=rhs)
            ast.copy_location(new_assign, node)
            self.dataframe_vars.update(new_df_vars)
            return new_assign
        # 3. Fallback: use astroid for ambiguous/complex cases
        result = self.generic_visit(node)
        return result

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        logger.debug(f"visit_Dict: keys={node.keys}, values={node.values}")
        new_keys = [self.visit(key) for key in node.keys]
        new_values = [self.visit(value) for value in node.values]
        logger.debug(f"visit_Dict: transformed keys={new_keys}, values={new_values}")
        new_dict = ast.Dict(keys=new_keys, values=new_values)
        ast.copy_location(new_dict, node)
        return new_dict

    def _extract_columns_from_node(self, node: ast.AST) -> List[str]:
        """Extract column names from an AST node (list, string constant, etc.)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        elif isinstance(node, ast.List):
            result = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    result.append(elt.value)
            return result
        return []
    
    def _extract_groupby_keys(self, node: ast.Call) -> List[str]:
        """Extract groupby keys from a groupby() call node."""
        keys = []
        
        if not isinstance(node, ast.Call):
            return keys
            
        # Check positional args (first arg is 'by')
        if node.args:
            keys = self._extract_columns_from_node(node.args[0])
        
        # Check keyword args
        for kw in node.keywords:
            if kw.arg == 'by':
                keys = self._extract_columns_from_node(kw.value)
                break
        
        return keys
    
    def _extract_columns_from_selection(self, node: ast.AST) -> List[str]:
        """Extract column names from a DataFrame selection like df[['A', 'B']]."""
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.List):
            return self._extract_columns_from_node(node.slice)
        return []
    
    def _get_df_var_from_node(self, node: ast.AST) -> Optional[str]:
        """Extract DataFrame variable name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return node.func.value.id
        return None

    def _get_target_name(self, node):
        """Extract the target name for assignment from a node.
        
        For simple variables like 'df', returns 'df_pl' if rename_dataframe is True.
        For attribute access like 'data.df', returns 'data.df_pl' if rename_dataframe is True.
        """
        if isinstance(node, ast.Name):
            # Simple variable: df -> df_pl
            return f"{node.id}_pl" if self.config.get('rename_dataframe', False) else node.id
        elif isinstance(node, ast.Attribute):
            # Attribute access: data.df -> data.df_pl
            base = self._get_target_name(node.value)
            if self.config.get('rename_dataframe', False):
                return f"{base}.{node.attr}_pl"
            else:
                return f"{base}.{node.attr}"
        else:
            # Fallback for complex expressions
            logger.warning(f"Complex expression in inplace operation: {ast.dump(node)}")
            return "result_pl" if self.config.get('rename_dataframe', False) else "result"

    def _extract_columns_from_df_creation(self, node: ast.AST) -> List[str]:
        """Extract column names from DataFrame creation nodes."""
        columns = []
        
        # Case: pl.DataFrame({'col1': values, 'col2': values})
        if (isinstance(node, ast.Call) and 
            isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and 
            node.func.value.id == 'pl' and 
            node.func.attr == 'DataFrame'):
            
            # Look for dictionary argument
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    for key in arg.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            columns.append(key.value)
            
            # Look for keyword arguments
            for kw in node.keywords:
                if kw.arg == 'data' and isinstance(kw.value, ast.Dict):
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            columns.append(key.value)
        
        # Case: pl.read_csv('file.csv')
        # In this case, we can't easily determine columns without actually reading the file
        
        return columns

def rewrite_chain_base_to_polars(node, pandas_aliases):
    """
    Recursively rewrite the base of a method chain if it is a DataFrame-creating function (e.g., pd.read_csv).
    Returns a new AST node with the base rewritten, or None if not applicable.
    """
    # If node is a call to pd.read_csv or pd.DataFrame
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id in pandas_aliases and node.func.attr in {'read_csv', 'DataFrame'}:
            # Rewrite to pl.read_csv or pl.DataFrame
            new_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='pl', ctx=ast.Load()),
                    attr=node.func.attr,
                    ctx=ast.Load()
                ),
                args=node.args,
                keywords=node.keywords
            )
            ast.copy_location(new_call, node)
            return new_call
    # If node is a method chain, recursively rewrite the base
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = node.func.value
        rewritten_base = rewrite_chain_base_to_polars(base, pandas_aliases)
        if rewritten_base is not None:
            # Rebuild the call with the rewritten base
            new_func = ast.Attribute(
                value=rewritten_base,
                attr=node.func.attr,
                ctx=ast.Load()
            )
            ast.copy_location(new_func, node.func)
            new_call = ast.Call(
                func=new_func,
                args=node.args,
                keywords=node.keywords
            )
            ast.copy_location(new_call, node)
            return new_call
    return None

# Utility: astroid fallback for DataFrame detection (stub for now)
def is_dataframe_var_astroid(var_name, assign_node):
    if not ASTROID_AVAILABLE:
        return False
    # This is a stub. Real implementation would use astroid to infer type.
    # For now, always return False.
    return False

# Utility: detect direct pandas DataFrame-creating function calls
def is_pandas_dataframe_creator_call(node, pandas_aliases):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in pandas_aliases
        and node.func.attr in {'read_csv', 'DataFrame'}
    )

# Utility: detect method call on known DataFrame variable
def is_method_call_on_known_df(node, known_df_vars):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in known_df_vars
    )
    