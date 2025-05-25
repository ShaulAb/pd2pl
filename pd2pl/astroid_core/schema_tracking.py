"""
Schema tracking for astroid-based DataFrame operations.

This module provides enhanced schema tracking capabilities using astroid's
type inference system, allowing for better understanding of DataFrame
transformations, especially in complex method chains.
"""

import astroid
from astroid import nodes
from typing import Optional, Dict, List, Set, Union, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from loguru import logger

# logger = logger.bind(name="AstroidSchemaState")

@dataclass
class AstroidSchemaState:
    """
    Track the schema state of a DataFrame through transformations.
    
    This class provides enhanced schema tracking by leveraging astroid's
    parent-child relationships and type inference capabilities.
    
    Attributes:
        name (str): Name of the DataFrame variable
        columns (Set[str]): Set of column names in the current schema
        group_keys (Set[str]): Columns used as group keys in groupby operations
        column_origins (Dict[str, str]): Mapping of derived columns to their source columns
        aggregated_columns (Dict[str, str]): Mapping of original columns to their aggregated versions
        column_types (Dict[str, str]): Maps column names to their inferred types
        in_groupby_chain (bool): Whether the current operation is part of a groupby chain
        _type_registry (Dict): Internal registry for type inference
    """
    name: str
    columns: Set[str] = field(default_factory=set)
    group_keys: Set[str] = field(default_factory=set)
    column_origins: Dict[str, str] = field(default_factory=dict)
    aggregated_columns: Dict[str, str] = field(default_factory=dict)
    column_types: Dict[str, str] = field(default_factory=dict)
    in_groupby_chain: bool = False
    _type_registry: Dict = field(
        default_factory=lambda: {
            'int': 'int64',
            'float': 'float64',
            'str': 'str',
            'bool': 'bool',
            'datetime64[ns]': 'datetime64[ns]',
            'timedelta64[ns]': 'timedelta64[ns]'
        }
    )
    
    def apply_groupby(self, group_keys: List[str]) -> 'AstroidSchemaState':
        """
        Apply groupby operation to schema.
        
        Args:
            group_keys: List of column names used as groupby keys
            
        Returns:
            Self, for method chaining
        """
        logger.debug(f"AstroidSchemaState.apply_groupby: original group_keys={self.group_keys}, new keys={group_keys}")
        self.in_groupby_chain = True
        
        # Add group keys to columns if not already present
        for key in group_keys:
            self.columns.add(key)
            
        # Update group keys
        self.group_keys.update(group_keys)
        logger.debug(f"AstroidSchemaState.apply_groupby: updated group_keys={self.group_keys}")
        return self
        
    def apply_aggregation(self, agg_function: str) -> 'AstroidSchemaState':
        """
        Apply aggregation to schema (after groupby).
        
        Args:
            agg_function: Name of the aggregation function (e.g., 'mean', 'sum')
            
        Returns:
            Self, for method chaining
        """
        if not self.in_groupby_chain:
            logger.debug("AstroidSchemaState.apply_aggregation: Not in groupby chain, skipping")
            return self
            
        logger.debug(f"AstroidSchemaState.apply_aggregation: applying {agg_function} to columns={self.columns}")
        new_columns = set(self.group_keys)  # Group keys remain
        
        # For each non-group column, create an aggregated version
        for col in self.columns:
            if col not in self.group_keys:
                agg_col = f"{col}_{agg_function}"
                new_columns.add(agg_col)
                # Store mapping from original to suffixed name
                self.aggregated_columns[col] = agg_col
                # Also store reverse mapping from suffixed to original name (for rename)
                self.column_origins[agg_col] = col
                
                # Update type information for the aggregated column
                if col in self.column_types:
                    # For numeric aggregations, the type usually stays numeric
                    # For other types, might need more sophisticated logic depending on the aggregation
                    self.column_types[agg_col] = self._infer_aggregated_type(self.column_types[col], agg_function)
                
                logger.debug(f"AstroidSchemaState.apply_aggregation: Mapped {col} -> {agg_col}")
        
        # Update the columns set with the new aggregated column names
        self.columns = new_columns
        logger.debug(f"AstroidSchemaState.apply_aggregation: Updated columns={self.columns}")
        logger.debug(f"AstroidSchemaState.apply_aggregation: Column mappings={self.aggregated_columns}")
        return self
    
    def infer_column_type(self, node: astroid.NodeNG, column_name: str) -> Optional[str]:
        """
        Infer the type of a column based on its usage in the AST.
        
        Args:
            node: The AST node where the column is used
            column_name: Name of the column to infer type for
            
        Returns:
            Inferred type as a string, or None if type cannot be determined
        """
        try:
            # First check if we already have type information
            if column_name in self.column_types:
                return self.column_types[column_name]
                
            # Try to infer type from the node's context
            inferred = None
            
            # Handle different node types that might contain type information
            if isinstance(node, astroid.Call) and hasattr(node.func, 'attrname'):
                # Method calls like df.column.astype('int')
                if node.func.attrname == 'astype' and node.args:
                    type_arg = node.args[0]
                    if isinstance(type_arg, astroid.Const):
                        return str(type_arg.value).lower()
                        
            elif isinstance(node, astroid.BinOp):
                # Binary operations can help infer numeric types
                left_type = self.infer_column_type(node.left, column_name) if hasattr(node, 'left') else None
                right_type = self.infer_column_type(node.right, column_name) if hasattr(node, 'right') else None
                
                # If both operands are numeric, result is numeric
                if left_type and right_type and \
                   'int' in left_type and 'int' in right_type:
                    return 'int64' if '64' in left_type or '64' in right_type else 'int32'
                elif left_type and 'float' in left_type or right_type and 'float' in right_type:
                    return 'float64'
                    
            elif isinstance(node, astroid.Compare):
                # Comparisons typically don't change the underlying type
                return self.infer_column_type(node.left, column_name) if hasattr(node, 'left') else None
                
            # If we couldn't infer from context, return a default based on column name
            if column_name.endswith(('_id', '_count', '_idx')):
                return 'int64'
            elif column_name.endswith(('_flag', 'is_', 'has_')):
                return 'bool'
                
            return None
            
        except Exception as e:
            logger.debug(f"Error inferring type for column {column_name}: {e}")
            return None
    
    def _infer_aggregated_type(self, original_type: str, agg_function: str) -> str:
        """
        Infer the type of an aggregated column based on the original type and aggregation function.
        
        Args:
            original_type: The type of the original column
            agg_function: The aggregation function being applied (e.g., 'mean', 'sum')
            
        Returns:
            The inferred type of the aggregated column
        """
        # Default to the original type if we can't determine a better one
        if not original_type:
            return 'float64'  # Most aggregations result in float by default
            
        # Handle common aggregation functions
        agg_function = agg_function.lower()
        original_type_lower = original_type.lower()
        
        # Handle boolean types first
        if 'bool' in original_type_lower:
            if agg_function in ('any', 'all'):
                return 'bool'
            elif agg_function in ('sum', 'count'):
                return 'int64'
            elif agg_function == 'mean':
                return 'float64'  # Proportion of True values
        
        # Aggregations that typically preserve the input type
        if agg_function in ('sum', 'min', 'max'):
            # These typically preserve the input type, but sum might promote to float
            if agg_function == 'sum' and 'int' in original_type_lower:
                return 'int64'  # Could potentially overflow, but default to int64
            return original_type
            
        # Aggregations that typically result in float
        elif agg_function in ('mean', 'std', 'var', 'median'):
            return 'float64'
            
        # Count-like aggregations
        elif agg_function in ('count', 'size'):
            return 'int64'
            
        # Boolean aggregations
        elif agg_function in ('any', 'all'):
            return 'bool'
            
        # First/last operations preserve the original type
        elif agg_function in ('first', 'last'):
            return original_type
            
        # For unknown aggregations, return a sensible default based on the original type
        if 'int' in original_type_lower or 'float' in original_type_lower:
            return 'float64'
            
        # If we can't determine, assume the type doesn't change
        return original_type
        
    def get_pandas_compat_rename_map(self) -> Dict[str, str]:
        """
        Get a mapping for renaming suffixed columns back to pandas-compatible names.
        
        Returns:
            Dict mapping from suffixed names to original names
        """
        rename_map = {agg_col: orig_col for orig_col, agg_col in self.aggregated_columns.items()}
        logger.debug(f"AstroidSchemaState.get_pandas_compat_rename_map: generated rename map {rename_map}")
        return rename_map

    def infer_columns_from_node(self, node: nodes.NodeNG) -> None:
        """Infer columns from an astroid node using type inference.
        
        This is a key advantage of using astroid - we can infer column names
        from context rather than relying on explicit references.
        
        Args:
            node: The astroid node to analyze
        """
        try:
            # Infer columns from DataFrame creation
            if isinstance(node, nodes.Call) and hasattr(node, 'func'):
                func_name = node.func.as_string() if hasattr(node.func, 'as_string') else str(node.func)
                if 'DataFrame' in func_name:
                    # Try to extract columns from a DataFrame constructor
                    self._infer_columns_from_dataframe_constructor(node)
                    
            # Infer columns from column selection
            elif isinstance(node, nodes.Subscript):
                # Handle df['column'] or df[['col1', 'col2']]
                self._infer_columns_from_subscript(node)
                
        except Exception as e:
            logger.error(f"Error inferring columns from node: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def _infer_columns_from_dataframe_constructor(self, node: nodes.Call) -> None:
        """
        Extract column names from a DataFrame constructor call.
        
        Args:
            node: The astroid Call node representing a DataFrame constructor
        """
        # Find the 'data' or first argument
        data_arg = None
        
        # Check keyword arguments
        if hasattr(node, 'keywords'):
            for keyword in node.keywords:
                if keyword.arg == 'data':
                    data_arg = keyword.value
                    break
        
        # If no data keyword, check positional args
        if data_arg is None and hasattr(node, 'args') and node.args:
            data_arg = node.args[0]
            
        if data_arg is None:
            logger.debug("No data argument found in DataFrame constructor")
            return
            
        # If data is a dict, extract keys as column names
        if isinstance(data_arg, nodes.Dict):
            if hasattr(data_arg, 'items'):
                # Try using items() for newer astroid versions
                for key, value in data_arg.items:
                    if isinstance(key, nodes.Const) and isinstance(key.value, str):
                        col_name = key.value
                        self.columns.add(col_name)
                        
                        # Add type inference for the column value
                        self._infer_column_type(col_name, value)
                        
                        logger.debug(f"Inferred column from DataFrame constructor items: {col_name}")
            elif hasattr(data_arg, 'keys'):
                # Try the keys attribute directly
                for i, key in enumerate(data_arg.keys):
                    if isinstance(key, nodes.Const) and isinstance(key.value, str):
                        col_name = key.value
                        self.columns.add(col_name)
                        
                        # Try to get the corresponding value if possible
                        if hasattr(data_arg, 'values') and i < len(data_arg.values):
                            value = data_arg.values[i]
                            self._infer_column_type(col_name, value)
                            
                        logger.debug(f"Inferred column from DataFrame constructor keys: {col_name}")
        
        # Also check for dtype information in the constructor
        self._infer_dtypes_from_constructor(node)
        
        logger.debug(f"After inference, columns = {self.columns}")
        logger.debug(f"After inference, column_types = {self.column_types}")
    
    def _infer_dtypes_from_constructor(self, node: nodes.Call) -> None:
        """
        Extract dtype information from a DataFrame constructor call.
        
        Args:
            node: The astroid Call node representing a DataFrame constructor
        """
        # Check for dtype argument
        if hasattr(node, 'keywords'):
            for keyword in node.keywords:
                if keyword.arg == 'dtype' or keyword.arg == 'dtypes':
                    dtype_arg = keyword.value
                    
                    # Handle dict of column -> dtype
                    if isinstance(dtype_arg, nodes.Dict):
                        if hasattr(dtype_arg, 'items'):
                            for key, value in dtype_arg.items:
                                if isinstance(key, nodes.Const) and isinstance(key.value, str):
                                    col_name = key.value
                                    if isinstance(value, nodes.Const):
                                        self.column_types[col_name] = self._normalize_dtype_str(value.value)
                                    elif hasattr(value, 'as_string'):
                                        self.column_types[col_name] = self._normalize_dtype_str(value.as_string())
                    
                    # Handle single dtype for all columns
                    elif isinstance(dtype_arg, nodes.Const):
                        dtype_value = self._normalize_dtype_str(dtype_arg.value)
                        # Apply to all known columns
                        for col in self.columns:
                            self.column_types[col] = dtype_value
                    
                    logger.debug(f"Inferred dtypes from constructor: {self.column_types}")
                    break
    
    def _normalize_dtype_str(self, dtype_value: Any) -> str:
        """
        Normalize dtype value to a string representation.
        
        Args:
            dtype_value: The dtype value, which could be a string, numpy type, etc.
            
        Returns:
            Normalized string representation of the dtype
        """
        if dtype_value is None:
            return 'unknown'
            
        if isinstance(dtype_value, str):
            # Already a string, normalize common representations
            dtype_str = dtype_value.lower()
            # Map common pandas/numpy dtype strings to simplified types
            if dtype_str in ['int8', 'int16', 'int32', 'int64', 'int', 'integer']:
                return 'int'
            elif dtype_str in ['float8', 'float16', 'float32', 'float64', 'float', 'double']:
                return 'float'
            elif dtype_str in ['bool', 'boolean']:
                return 'bool'
            elif dtype_str in ['str', 'string', 'object']:
                return 'str'
            elif dtype_str in ['datetime64', 'datetime', 'date']:
                return 'datetime'
            elif dtype_str in ['timedelta64', 'timedelta']:
                return 'timedelta'
            elif dtype_str in ['category', 'categorical']:
                return 'category'
            else:
                return dtype_str
        else:
            # For non-string values, just convert to string
            return str(dtype_value)
    
    def _infer_column_type(self, column_name: str, value_node: nodes.NodeNG) -> None:
        """
        Infer the type of a column from its value expression.
        
        Args:
            column_name: Name of the column
            value_node: Astroid node representing the column value
        """
        try:
            # Use astroid's type inference
            inferred_types = set()
            
            # For list/array values, look at the first element if possible
            if isinstance(value_node, nodes.List) and hasattr(value_node, 'elts') and value_node.elts:
                first_element = value_node.elts[0]
                self._infer_from_value(first_element, inferred_types)
            else:
                # Try direct inference on the value node
                self._infer_from_value(value_node, inferred_types)
            
            # If we have inferred types, use the first one (or a merged representation)
            if inferred_types:
                self.column_types[column_name] = self._merge_inferred_types(inferred_types)
                logger.debug(f"Inferred type for column {column_name}: {self.column_types[column_name]}")
            else:
                # If inference failed, try to determine based on node type
                if isinstance(value_node, nodes.List):
                    self.column_types[column_name] = 'list'
                elif isinstance(value_node, nodes.Dict):
                    self.column_types[column_name] = 'dict'
                else:
                    self.column_types[column_name] = 'unknown'
                logger.debug(f"Couldn't infer type for {column_name}, defaulting to {self.column_types[column_name]}")
                
        except Exception as e:
            logger.error(f"Error inferring type for column {column_name}: {str(e)}")
            self.column_types[column_name] = 'unknown'
    
    def _infer_from_value(self, node: nodes.NodeNG, inferred_types: Set[str]) -> None:
        """
        Infer types from a value node and add them to the inferred_types set.
        
        Args:
            node: Astroid node to infer from
            inferred_types: Set to add inferred types to
        """
        # Direct type inference for constants
        if isinstance(node, nodes.Const):
            if node.value is None:
                inferred_types.add('NoneType')
            elif isinstance(node.value, int):
                # Special case for boolean values which are also instances of int in Python
                if isinstance(node.value, bool):
                    inferred_types.add('bool')
                else:
                    inferred_types.add('int')
            elif isinstance(node.value, float):
                inferred_types.add('float')
            elif isinstance(node.value, str):
                inferred_types.add('str')
            elif isinstance(node.value, bool):
                inferred_types.add('bool')
            else:
                inferred_types.add(type(node.value).__name__)
            return
            
        # Use astroid's infer() method for more complex expressions
        try:
            inferred = node.infer()
            for inf in inferred:
                if inf is astroid.Uninferable:
                    continue
                    
                # Get type from inferred object
                if hasattr(inf, 'name'):
                    inferred_types.add(inf.name)
                elif hasattr(inf, 'pytype'):
                    pytype = inf.pytype()
                    # Convert pytype() result (e.g., 'builtins.int') to simple type name
                    if '.' in pytype:
                        inferred_types.add(pytype.split('.')[-1])
                    else:
                        inferred_types.add(pytype)
                else:
                    inferred_types.add(type(inf).__name__)
        except Exception:
            # If inference fails, we don't add any types
            pass
    
    def _merge_inferred_types(self, inferred_types: Set[str]) -> str:
        """
        Merge multiple inferred types into a single representation.
        
        Args:
            inferred_types: Set of inferred type strings
            
        Returns:
            Merged type representation
        """
        # Simple priority-based merging
        if not inferred_types:
            return 'unknown'
            
        # Normalize the types
        normalized_types = {self._normalize_dtype_str(t) for t in inferred_types}
        
        # If there's only one type, return it
        if len(normalized_types) == 1:
            return next(iter(normalized_types))
            
        # If there are multiple types, try to find a common denominator
        numeric_types = {'int', 'float', 'complex'}
        if normalized_types.issubset(numeric_types):
            # For mixed numeric types, use the most general one
            if 'complex' in normalized_types:
                return 'complex'
            elif 'float' in normalized_types:
                return 'float'
            else:
                return 'int'
        
        # If we have a mix of types (like int and str), use 'object' as the common type
        # This matches pandas behavior where mixed types become object dtype
        if len(normalized_types) > 1 and not normalized_types.issubset(numeric_types):
            return 'object'
                
        # Otherwise, join the types with a '|' to indicate a union
        return '|'.join(sorted(normalized_types))
    
    def _infer_columns_from_subscript(self, node: nodes.Subscript) -> None:
        """
        Extract column names from a subscript operation like df['column'].
        
        Args:
            node: The astroid Subscript node
        """
        try:
            # Different versions of astroid might have different attribute names
            # or structures for the slice
            
            # First try to get slice directly
            if hasattr(node, 'slice'):
                slice_value = node.slice
                
                # Handle simple string case: df['column']
                if isinstance(slice_value, nodes.Const) and isinstance(slice_value.value, str):
                    col_name = slice_value.value
                    self.columns.add(col_name)
                    logger.debug(f"Inferred column from subscript: {col_name}")
                    
                # Handle list case: df[['col1', 'col2']]
                elif isinstance(slice_value, nodes.List):
                    if hasattr(slice_value, 'elts'):
                        for elt in slice_value.elts:
                            if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                                col_name = elt.value
                                self.columns.add(col_name)
                                logger.debug(f"Inferred column from subscript list: {col_name}")
            
            # For older astroid versions, might need to access differently
            elif hasattr(node, 'value') and hasattr(node, 'slice'):
                # In older versions, slice might be accessible through a different path
                slice_value = node.slice
                
                if isinstance(slice_value, nodes.Index):
                    # In some versions, slice is wrapped in an Index node
                    value = slice_value.value
                    if isinstance(value, nodes.Const) and isinstance(value.value, str):
                        col_name = value.value
                        self.columns.add(col_name)
                        logger.debug(f"Inferred column from Index subscript: {col_name}")
                    elif isinstance(value, nodes.List):
                        for elt in value.elts:
                            if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                                col_name = elt.value
                                self.columns.add(col_name)
                                logger.debug(f"Inferred column from Index subscript list: {col_name}")
                                
            logger.debug(f"After subscript inference, columns = {self.columns}")
                                
        except Exception as e:
            logger.error(f"Error inferring columns from subscript: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
    def update_column_types(self, column_types: Dict[str, str]) -> None:
        """
        Update column types with new type information.
        
        Args:
            column_types: Dictionary mapping column names to their types
        """
        self.column_types.update(column_types)
        logger.debug(f"Updated column_types: {self.column_types}")
    
    def get_column_type(self, column_name: str) -> str:
        """
        Get the type of a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Type of the column as a string, or 'unknown' if not found
        """
        return self.column_types.get(column_name, 'unknown')
    
    def copy(self) -> 'AstroidSchemaState':
        """
        Create a deep copy of this schema state.
        
        Returns:
            A new AstroidSchemaState with the same data
        """
        new_state = AstroidSchemaState(name=self.name)
        new_state.columns = set(self.columns)
        new_state.group_keys = set(self.group_keys)
        new_state.column_origins = dict(self.column_origins)
        new_state.aggregated_columns = dict(self.aggregated_columns)
        new_state.column_types = dict(self.column_types)
        new_state.in_groupby_chain = self.in_groupby_chain
        return new_state

class AstroidSchemaRegistry:
    """
    Registry for DataFrame schemas using astroid's enhanced type system.
    """
    def __init__(self):
        self.schemas: Dict[str, AstroidSchemaState] = {}
        
    def register_dataframe(self, name: str, columns: Optional[List[str]] = None, 
                         column_types: Optional[Dict[str, str]] = None) -> AstroidSchemaState:
        """
        Register a new DataFrame in the registry.
        
        Args:
            name: Name of the DataFrame variable
            columns: Optional list of initial column names
            column_types: Optional dictionary mapping column names to their types
            
        Returns:
            The created schema state
        """
        logger.debug(f"AstroidSchemaRegistry.register_dataframe: {name} with columns={columns}, types={column_types}")
        schema = AstroidSchemaState(name=name)
        if columns:
            schema.columns.update(columns)
        if column_types:
            schema.column_types.update(column_types)
        self.schemas[name] = schema
        return schema
        
    def get_schema(self, name: str) -> Optional[AstroidSchemaState]:
        """
        Get the schema for a DataFrame by name.
        
        Args:
            name: Name of the DataFrame variable
            
        Returns:
            The schema state, or None if not found
        """
        return self.schemas.get(name)
        
    def update_schema(self, name: str, schema: AstroidSchemaState) -> None:
        """
        Update the schema for a DataFrame.
        
        Args:
            name: Name of the DataFrame variable
            schema: The updated schema state
        """
        self.schemas[name] = schema
        logger.debug(f"AstroidSchemaRegistry.update_schema: updated {name}")
    
    def copy_schema(self, from_name: str, to_name: str) -> Optional[AstroidSchemaState]:
        """
        Copy a schema from one DataFrame to another.
        
        This is useful for handling DataFrame assignments like df2 = df1.
        
        Args:
            from_name: Source DataFrame name
            to_name: Target DataFrame name
            
        Returns:
            The copied schema, or None if source not found
        """
        source_schema = self.get_schema(from_name)
        if source_schema is None:
            logger.debug(f"AstroidSchemaRegistry.copy_schema: source {from_name} not found")
            return None
            
        copied_schema = source_schema.copy()
        copied_schema.name = to_name
        self.schemas[to_name] = copied_schema
        logger.debug(f"AstroidSchemaRegistry.copy_schema: copied {from_name} to {to_name}")
        return copied_schema
