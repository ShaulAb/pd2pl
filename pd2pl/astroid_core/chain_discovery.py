"""
Chain discovery module for identifying and analyzing pandas method chains.

This module provides the core infrastructure for detecting method chains in pandas code
and preparing them for transformation to polars equivalents.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import astroid
from astroid import nodes

# Import schema tracking for integration
from .schema_tracking import AstroidSchemaState, AstroidSchemaRegistry


class ChainType(Enum):
    """Types of method chains."""
    FILTER = "filter"
    TRANSFORM = "transform"
    AGGREGATION = "aggregation"
    MIXED = "mixed"


@dataclass
class MethodCall:
    """Represents a single method call in a chain."""
    name: str
    node: nodes.NodeNG
    args: List[nodes.NodeNG]
    kwargs: dict


@dataclass
class MethodChain:
    """Represents a complete method chain."""
    start_node: nodes.NodeNG
    end_node: nodes.NodeNG
    methods: List[MethodCall]
    variable_name: Optional[str] = None
    chain_type: ChainType = ChainType.MIXED
    complexity_score: float = 0.0
    dataframe_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Schema tracking integration
    input_schema: Optional[AstroidSchemaState] = None
    output_schema: Optional[AstroidSchemaState] = None
    schema_changes: List[str] = field(default_factory=list)


class ChainRegistry:
    """Registry for storing and managing discovered method chains."""
    
    def __init__(self, schema_registry: Optional[AstroidSchemaRegistry] = None):
        self.chains: List[MethodChain] = []
        self._chains_by_type: Dict[ChainType, List[MethodChain]] = {
            chain_type: [] for chain_type in ChainType
        }
        self.schema_registry = schema_registry or AstroidSchemaRegistry()
    
    def register_chain(self, chain: MethodChain) -> None:
        """
        Register a method chain in the registry.
        
        Args:
            chain: The method chain to register
        """
        # Classify and score the chain
        chain.chain_type = self._classify_chain(chain)
        chain.complexity_score = self._calculate_complexity(chain)
        chain.dataframe_reference = self._extract_dataframe_reference(chain)
        
        # Track schema changes through the chain
        self._track_schema_changes(chain)
        
        # Add to registry
        self.chains.append(chain)
        self._chains_by_type[chain.chain_type].append(chain)
    
    def get_all_chains(self) -> List[MethodChain]:
        """Get all registered chains."""
        return self.chains.copy()
    
    def get_chains_by_type(self, chain_type: ChainType) -> List[MethodChain]:
        """Get all chains of a specific type."""
        return self._chains_by_type[chain_type].copy()
    
    def get_chains_by_complexity(self, min_score: float = 0.0, max_score: float = float('inf')) -> List[MethodChain]:
        """Get chains within a complexity score range."""
        return [chain for chain in self.chains 
                if min_score <= chain.complexity_score <= max_score]
    
    def get_chains_by_dataframe(self, dataframe_name: str) -> List[MethodChain]:
        """Get all chains that operate on a specific DataFrame."""
        return [chain for chain in self.chains 
                if chain.dataframe_reference == dataframe_name]
    
    def get_chains_with_schema_changes(self) -> List[MethodChain]:
        """Get all chains that modify the schema."""
        return [chain for chain in self.chains if chain.schema_changes]
    
    def get_chains_by_schema_change_type(self, change_pattern: str) -> List[MethodChain]:
        """Get chains that contain a specific type of schema change."""
        return [chain for chain in self.chains 
                if any(change_pattern.lower() in change.lower() for change in chain.schema_changes)]
    
    def get_schema_evolution(self, dataframe_name: str) -> List[AstroidSchemaState]:
        """Get the schema evolution for a specific DataFrame through all chains."""
        evolution = []
        chains = self.get_chains_by_dataframe(dataframe_name)
        
        for chain in sorted(chains, key=lambda c: c.start_node.lineno if hasattr(c.start_node, 'lineno') else 0):
            if chain.input_schema:
                evolution.append(chain.input_schema)
            if chain.output_schema:
                evolution.append(chain.output_schema)
        
        return evolution
    
    def _classify_chain(self, chain: MethodChain) -> ChainType:
        """Classify a chain based on its method types."""
        method_names = [method.name for method in chain.methods]
        
        # Define method categories
        filter_methods = {'filter', '__getitem__', 'query', 'where'}
        transform_methods = {'rename', 'sort_values', 'drop', 'dropna', 'fillna', 'assign', 'with_columns'}
        aggregation_methods = {'groupby', 'agg', 'mean', 'sum', 'count', 'max', 'min', 'std', 'var'}
        
        has_filter = any(method in filter_methods for method in method_names)
        has_transform = any(method in transform_methods for method in method_names)
        has_aggregation = any(method in aggregation_methods for method in method_names)
        
        # Classify based on combination
        if has_aggregation and (has_filter or has_transform):
            return ChainType.MIXED
        elif has_aggregation:
            return ChainType.AGGREGATION
        elif has_filter and has_transform:
            return ChainType.MIXED
        elif has_filter:
            return ChainType.FILTER
        elif has_transform:
            return ChainType.TRANSFORM
        else:
            return ChainType.MIXED
    
    def _calculate_complexity(self, chain: MethodChain) -> float:
        """Calculate complexity score for a chain."""
        base_score = len(chain.methods)
        
        # Add complexity for specific patterns
        complexity_bonus = 0.0
        
        for method in chain.methods:
            # Boolean indexing adds complexity
            if method.name == '__getitem__':
                complexity_bonus += 0.5
            
            # Groupby operations add complexity
            if method.name == 'groupby':
                complexity_bonus += 1.0
            
            # Complex aggregations add complexity
            if method.name == 'agg' and (method.kwargs or method.args):
                complexity_bonus += 0.5
        
        return base_score + complexity_bonus
    
    def _extract_dataframe_reference(self, chain: MethodChain) -> Optional[str]:
        """Extract the DataFrame variable name from the chain."""
        if isinstance(chain.start_node, nodes.Name):
            return chain.start_node.name
        return None
    
    def _track_schema_changes(self, chain: MethodChain) -> None:
        """
        Track schema changes through a method chain.
        
        Args:
            chain: The method chain to track schema changes for
        """
        if not chain.dataframe_reference:
            return
        
        # Get or create input schema
        input_schema = self.schema_registry.get_schema(chain.dataframe_reference)
        if input_schema is None:
            # Create a basic schema if none exists
            input_schema = self.schema_registry.register_dataframe(chain.dataframe_reference)
            # Note: This creates an empty schema, which might not be what we want
        
        # Copy input schema as starting point
        chain.input_schema = input_schema.copy()
        current_schema = input_schema.copy()
        
        # Apply each method in the chain to track schema evolution
        for method in chain.methods:
            schema_change = self._apply_method_to_schema(current_schema, method)
            if schema_change:
                chain.schema_changes.append(schema_change)
        
        # Store final schema state
        chain.output_schema = current_schema
        
        # Update the schema registry with the final state if this is an assignment
        if chain.variable_name:
            # Create a copy with the correct name for the new variable
            final_schema = current_schema.copy()
            final_schema.name = chain.variable_name
            self.schema_registry.update_schema(chain.variable_name, final_schema)
    
    def _apply_method_to_schema(self, schema: AstroidSchemaState, method: MethodCall) -> Optional[str]:
        """
        Apply a single method to a schema state and return description of change.
        
        Args:
            schema: The schema state to modify
            method: The method call to apply
            
        Returns:
            Description of the schema change, or None if no change
        """
        method_name = method.name
        
        if method_name == 'groupby':
            # Extract groupby keys
            group_keys = self._extract_groupby_keys(method)
            if group_keys:
                schema.apply_groupby(group_keys)
                return f"Applied groupby on columns: {group_keys}"
        
        elif method_name in ['mean', 'sum', 'count', 'max', 'min', 'std', 'var']:
            # Direct aggregation methods
            if schema.in_groupby_chain:
                schema.apply_aggregation(method_name)
                return f"Applied {method_name} aggregation"
        
        elif method_name == 'agg':
            # Complex aggregation
            agg_functions = self._extract_agg_functions(method)
            for func in agg_functions:
                schema.apply_aggregation(func)
            return f"Applied aggregation functions: {agg_functions}"
        
        elif method_name == '__getitem__':
            # Column selection or boolean indexing
            columns = self._extract_selected_columns(method)
            if columns:
                # Update schema to only include selected columns
                schema.columns = schema.columns.intersection(set(columns))
                return f"Selected columns: {columns}"
            else:
                return "Applied boolean indexing (no column changes)"
        
        elif method_name == 'rename':
            # Column renaming
            rename_map = self._extract_rename_mapping(method)
            if rename_map:
                # Apply renaming to schema
                new_columns = set()
                for col in schema.columns:
                    new_col = rename_map.get(col, col)
                    new_columns.add(new_col)
                schema.columns = new_columns
                return f"Renamed columns: {rename_map}"
        
        elif method_name in ['drop', 'dropna']:
            # Column dropping
            if method_name == 'drop':
                dropped_cols = self._extract_dropped_columns(method)
                if dropped_cols:
                    schema.columns = schema.columns - set(dropped_cols)
                    return f"Dropped columns: {dropped_cols}"
            return f"Applied {method_name}"
        
        elif method_name == 'reset_index':
            # Reset index (no-op for polars)
            return "Reset index (no-op for polars)"
        
        return None
    
    def _extract_groupby_keys(self, method: MethodCall) -> List[str]:
        """Extract groupby keys from a groupby method call."""
        keys = []
        
        # Check positional arguments
        for arg in method.args:
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
                keys.append(arg.value)
            elif isinstance(arg, nodes.List):
                for elt in arg.elts:
                    if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                        keys.append(elt.value)
        
        # Check keyword arguments
        if 'by' in method.kwargs:
            by_arg = method.kwargs['by']
            if isinstance(by_arg, nodes.Const) and isinstance(by_arg.value, str):
                keys.append(by_arg.value)
            elif isinstance(by_arg, nodes.List):
                for elt in by_arg.elts:
                    if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                        keys.append(elt.value)
        
        return keys
    
    def _extract_agg_functions(self, method: MethodCall) -> List[str]:
        """Extract aggregation functions from an agg method call."""
        functions = []
        
        # Check positional arguments
        for arg in method.args:
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
                functions.append(arg.value)
            elif isinstance(arg, nodes.List):
                for elt in arg.elts:
                    if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                        functions.append(elt.value)
            elif isinstance(arg, nodes.Dict):
                # Handle dict-style aggregation like {'col': 'mean'}
                for key, value in arg.items:
                    if isinstance(value, nodes.Const) and isinstance(value.value, str):
                        functions.append(value.value)
        
        return functions if functions else ['mean']  # Default to mean if no function specified
    
    def _extract_selected_columns(self, method: MethodCall) -> Optional[List[str]]:
        """Extract selected columns from a __getitem__ method call."""
        if not method.args:
            return None
        
        slice_arg = method.args[0]
        columns = []
        
        if isinstance(slice_arg, nodes.Const) and isinstance(slice_arg.value, str):
            columns.append(slice_arg.value)
        elif isinstance(slice_arg, nodes.List):
            for elt in slice_arg.elts:
                if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                    columns.append(elt.value)
        else:
            # Complex indexing (boolean, etc.) - return None to indicate no column selection
            return None
        
        return columns
    
    def _extract_rename_mapping(self, method: MethodCall) -> Dict[str, str]:
        """Extract column rename mapping from a rename method call."""
        rename_map = {}
        
        # Check positional arguments
        if method.args and isinstance(method.args[0], nodes.Dict):
            dict_arg = method.args[0]
            for key, value in dict_arg.items:
                if (isinstance(key, nodes.Const) and isinstance(key.value, str) and
                    isinstance(value, nodes.Const) and isinstance(value.value, str)):
                    rename_map[key.value] = value.value
        
        # Check keyword arguments
        if 'columns' in method.kwargs:
            columns_arg = method.kwargs['columns']
            if isinstance(columns_arg, nodes.Dict):
                for key, value in columns_arg.items:
                    if (isinstance(key, nodes.Const) and isinstance(key.value, str) and
                        isinstance(value, nodes.Const) and isinstance(value.value, str)):
                        rename_map[key.value] = value.value
        
        return rename_map
    
    def _extract_dropped_columns(self, method: MethodCall) -> List[str]:
        """Extract dropped columns from a drop method call."""
        columns = []
        
        # Check positional arguments
        for arg in method.args:
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
                columns.append(arg.value)
            elif isinstance(arg, nodes.List):
                for elt in arg.elts:
                    if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                        columns.append(elt.value)
        
        # Check keyword arguments
        if 'columns' in method.kwargs:
            columns_arg = method.kwargs['columns']
            if isinstance(columns_arg, nodes.Const) and isinstance(columns_arg.value, str):
                columns.append(columns_arg.value)
            elif isinstance(columns_arg, nodes.List):
                for elt in columns_arg.elts:
                    if isinstance(elt, nodes.Const) and isinstance(elt.value, str):
                        columns.append(elt.value)
        
        return columns


class ChainDiscovery:
    """Discovers and analyzes method chains in astroid AST."""
    
    def __init__(self, registry: Optional[ChainRegistry] = None, schema_registry: Optional[AstroidSchemaRegistry] = None):
        self.schema_registry = schema_registry or AstroidSchemaRegistry()
        self.registry = registry or ChainRegistry(self.schema_registry)
    
    def discover_chains(self, module: nodes.Module) -> List[MethodChain]:
        """
        Discover all method chains in the given astroid module.
        
        Args:
            module: The astroid module to analyze
            
        Returns:
            List of discovered method chains
        """
        chains = []
        
        # Find all assignment statements
        for node in module.nodes_of_class(nodes.Assign):
            if node.value:
                chain = self._build_chain_from_node(node.value)
                if chain:
                    # Extract variable name from assignment target
                    if node.targets and len(node.targets) > 0:
                        target = node.targets[0]
                        if isinstance(target, (nodes.Name, nodes.AssignName)):
                            chain.variable_name = target.name
                        else:
                            print(f"DEBUG: Target is not a Name/AssignName node: {type(target)}")
                    else:
                        print(f"DEBUG: No targets found in assignment")
                    
                    # Register the chain in the registry
                    self.registry.register_chain(chain)
                    chains.append(chain)
        
        return chains
    
    def _build_chain_from_node(self, node: nodes.NodeNG) -> Optional[MethodChain]:
        """
        Build a method chain starting from the given node.
        
        Args:
            node: The node to start building the chain from
            
        Returns:
            MethodChain if a chain is found, None otherwise
        """
        methods = []
        current = node
        start_node = node
        
        # Traverse the chain from right to left (end to start)
        while current:
            if isinstance(current, nodes.Call):
                # This is a method call
                if isinstance(current.func, nodes.Attribute):
                    # Check if this is a constructor call (like pd.DataFrame)
                    if self._is_constructor_call(current):
                        # Stop here - don't include constructor calls in method chains
                        break
                    
                    method_call = MethodCall(
                        name=current.func.attrname,
                        node=current,
                        args=current.args,
                        kwargs={kw.arg: kw.value for kw in current.keywords if kw.arg}
                    )
                    methods.append(method_call)
                    current = current.func.expr
                else:
                    break
            elif isinstance(current, nodes.Subscript):
                # This is a subscript operation like df[df.A > 1]
                method_call = MethodCall(
                    name="__getitem__",
                    node=current,
                    args=[current.slice],
                    kwargs={}
                )
                methods.append(method_call)
                current = current.value
            else:
                # End of chain - this should be the DataFrame variable
                break
        
        if methods:
            # Reverse methods to get them in execution order
            methods.reverse()
            return MethodChain(
                start_node=current if current else start_node,
                end_node=node,
                methods=methods
            )
        
        return None
    
    def _is_constructor_call(self, call_node: nodes.Call) -> bool:
        """
        Check if a call node represents a constructor call that should be excluded from chains.
        
        Args:
            call_node: The call node to check
            
        Returns:
            True if this is a constructor call, False otherwise
        """
        if not isinstance(call_node.func, nodes.Attribute):
            return False
        
        method_name = call_node.func.attrname
        
        # Common DataFrame/Series constructors
        constructor_names = {
            'DataFrame', 'Series', 'Index', 'MultiIndex',
            'read_csv', 'read_json', 'read_excel', 'read_parquet',
            'read_sql', 'read_html', 'read_pickle', 'read_feather'
        }
        
        # Check if this is a known constructor method
        if method_name in constructor_names:
            return True
        
        # Check if the object being called is a known pandas module
        if isinstance(call_node.func.expr, nodes.Name):
            module_name = call_node.func.expr.name
            # Only flag as constructor if called on known pandas module names
            pandas_modules = {'pd', 'pandas'}
            if module_name in pandas_modules:
                return True
        
        return False 