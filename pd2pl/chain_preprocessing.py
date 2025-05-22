"""Chain preprocessing module for identifying method chains before transformation.

This module provides functionality to:
1. Detect method chains in AST
2. Analyze schemas across chains
3. Prepare metadata for the transformer
"""
import ast
from typing import Dict, Set, List, Optional, Any, Tuple

from pd2pl.logging import logger
from pd2pl.chain_tracking import ChainRegistry, ChainNode
from pd2pl.schema_tracking import SchemaState, SchemaRegistry
from pd2pl.mapping import method_maps


class ChainDetectionVisitor(ast.NodeVisitor):
    """AST visitor for detecting method chains."""
    
    def __init__(self):
        self.chain_registry = ChainRegistry()
        self.dataframe_vars = set(['df', 'df_left', 'df_right'])
        
    def visit_Assign(self, node: ast.Assign) -> None:
        """Track DataFrame variables created in assignments."""
        if isinstance(node.value, ast.Call):
            # Basic check for DataFrame creation
            if self._is_dataframe_creation(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.dataframe_vars.add(target.id)
                        logger.debug(f"ChainDetection: Identified DataFrame variable: {target.id}")
        self.generic_visit(node)
        
    def _is_dataframe_creation(self, node: ast.Call) -> bool:
        """Check if a call creates a DataFrame."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in ('pd', 'pandas') and node.func.attr in ('DataFrame', 'read_csv', 'read_excel'):
                    return True
        return False
        
    def visit_Call(self, node: ast.Call) -> None:
        """Process calls to detect method chains."""
        if isinstance(node.func, ast.Attribute):
            # Method call pattern: object.method()
            method_name = node.func.attr
            
            # Check if this is a method call on a DataFrame or another call
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.dataframe_vars:
                # Direct method call on DataFrame: df.method()
                chain_node = self.chain_registry.register_node(node, method_name)
                self.chain_registry.add_to_chain(None, chain_node)  # Root node
                logger.debug(f"ChainDetection: Found root method: {method_name} on {node.func.value.id}")
                
            elif isinstance(node.func.value, ast.Call):
                # Method chain: something().method()
                parent_call = node.func.value
                chain_node = self.chain_registry.register_node(node, method_name)
                self.chain_registry.add_to_chain(parent_call, chain_node)
                logger.debug(f"ChainDetection: Found chained method: {method_name} on {ast.dump(parent_call, annotate_fields=False)[:50]}...")
                
                # Store this node with its parent to ensure we can build a complete chain
                if hasattr(self, 'chain_parents'):
                    self.chain_parents[id(node)] = parent_call
                else:
                    self.chain_parents = {id(node): parent_call}
                
            elif isinstance(node.func.value, ast.Subscript) and isinstance(node.func.value.value, ast.Name):
                # Column selection followed by method: df['col'].method() or df[['col1', 'col2']].method()
                df_name = node.func.value.value.id
                if df_name in self.dataframe_vars:
                    chain_node = self.chain_registry.register_node(node, method_name)
                    self.chain_registry.add_to_chain(None, chain_node)  # Root node (column reference)
                    
                    # Extract selected columns for better schema tracking
                    columns = self._extract_columns_from_selection(node.func.value)
                    col_str = ', '.join(columns) if columns else 'unknown'
                    logger.debug(f"ChainDetection: Found column selection method: {method_name} on {df_name}[{col_str}]")
        
        # Continue with all children
        self.generic_visit(node)
        
    def _extract_columns_from_selection(self, node: ast.Subscript) -> List[str]:
        """Extract column names from a DataFrame selection."""
        columns = []
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            # Single column: df['col']
            columns.append(node.slice.value)
        elif isinstance(node.slice, ast.List):
            # Multiple columns: df[['col1', 'col2']]
            for elt in node.slice.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    columns.append(elt.value)
        return columns


def preprocess_chains(tree: ast.AST, dataframe_vars: Set[str]) -> Tuple[ChainRegistry, SchemaRegistry]:
    """Preprocess AST to identify method chains and calculate schemas."""
    # Phase 1: Identify method chains
    chain_detector = ChainDetectionVisitor()
    chain_detector.visit(tree)
    chain_registry = chain_detector.chain_registry
    
    # Connect chains if we have parent information
    if hasattr(chain_detector, 'chain_parents'):
        logger.debug("Connecting chains based on parent-child relationships")
        # Find all root nodes that are actually part of a larger chain
        for node_id, node in chain_registry.nodes_by_id.items():
            if node.parent is None and id(node.node) in chain_detector.chain_parents:
                parent_node = chain_detector.chain_parents[id(node.node)]
                parent_chain_node = chain_registry.get_node(parent_node)
                if parent_chain_node:
                    logger.debug(f"Connecting {node.method_name} to parent {parent_chain_node.method_name}")
                    parent_chain_node.add_child(node)
                    # Remove from root nodes if it was there
                    if node in chain_registry.root_nodes:
                        chain_registry.root_nodes.remove(node)
    
    # Now finalize the chains
    chain_registry.finalize_chains()
    
    # Debug output of chains
    logger.debug("Chain preprocessing complete. Identified chains:")
    chain_registry.print_chains()
    
    # Phase 2: Analyze schemas through chains
    schema_registry = SchemaRegistry()
    analyze_chain_schemas(chain_registry, schema_registry, dataframe_vars)
    
    return chain_registry, schema_registry


def analyze_chain_schemas(chain_registry: ChainRegistry, schema_registry: SchemaRegistry, dataframe_vars: Set[str]) -> None:
    """Analyze schemas across method chains."""
    # Process chains in dependency order (roots first)
    for chain_id, nodes in chain_registry.chains_by_id.items():
        # Get root node and sort nodes by position
        sorted_nodes = sorted(nodes, key=lambda n: n.position)
        root_node = sorted_nodes[0]
        
        logger.debug(f"Schema analysis for chain {chain_id}: {' -> '.join(n.method_name for n in sorted_nodes)}")
        
        # Determine the DataFrame variable and starting schema
        df_var = None
        selected_columns = []
        
        # Case 1: Direct method on DataFrame
        if isinstance(root_node.node.func.value, ast.Name) and root_node.node.func.value.id in dataframe_vars:
            df_var = root_node.node.func.value.id
            
        # Case 2: Method on column selection
        elif (isinstance(root_node.node.func.value, ast.Subscript) and 
              isinstance(root_node.node.func.value.value, ast.Name) and
              root_node.node.func.value.value.id in dataframe_vars):
            
            df_var = root_node.node.func.value.value.id
            
            # Extract column selection
            subscript_node = root_node.node.func.value
            if isinstance(subscript_node.slice, ast.Constant) and isinstance(subscript_node.slice.value, str):
                selected_columns = [subscript_node.slice.value]
            elif isinstance(subscript_node.slice, ast.List):
                selected_columns = [
                    elt.value for elt in subscript_node.slice.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
        
        if not df_var:
            logger.debug(f"  Could not determine DataFrame for chain {chain_id}, skipping schema analysis")
            continue
            
        logger.debug(f"  Base DataFrame: {df_var}, selected columns: {selected_columns}")
        
        # Get or create schema
        schema = schema_registry.get_schema(df_var)
        if not schema:
            schema = schema_registry.register_dataframe(df_var)
            logger.debug(f"  Created new schema for {df_var}")
        else:
            logger.debug(f"  Using existing schema for {df_var}: columns={schema.columns}")
        
        # Start with a copy for this chain's context
        current_schema = schema.copy()
        
        # Apply column selection if present
        if selected_columns:
            current_schema.apply_selection(selected_columns)
            logger.debug(f"  Applied selection to schema: {selected_columns}")
        
        # Store initial schema in root node
        root_node.schema_before = current_schema.copy()
        
        # Process each node in the chain to update schemas
        for i, node in enumerate(sorted_nodes):
            # Update schema based on method
            updated_schema = _update_schema_for_method(node, current_schema)
            node.schema_after = updated_schema.copy()
            
            # Pass schema to next node in chain
            if i < len(sorted_nodes) - 1:
                next_node = sorted_nodes[i + 1]
                next_node.schema_before = updated_schema.copy()
                
            # Update current schema for next iteration
            current_schema = updated_schema
            
            logger.debug(f"  After {node.method_name}: columns={current_schema.columns}, "
                        f"groupby={current_schema.in_groupby_chain}, "
                        f"agg_cols={current_schema.aggregated_columns}")


def _update_schema_for_method(chain_node: ChainNode, schema: SchemaState) -> SchemaState:
    """Update schema based on method effects."""
    method_name = chain_node.method_name
    node = chain_node.node
    
    # Create a copy of the schema to update
    new_schema = schema.copy()
    
    # Extract arguments
    args = node.args
    kwargs = {kw.arg: kw.value for kw in node.keywords}
    
    # Handle specific methods
    if method_name == 'groupby':
        # Extract group keys
        group_keys = _extract_groupby_keys(node)
        if group_keys:
            new_schema.apply_groupby(group_keys)
            logger.debug(f"  Schema updated for groupby with keys: {group_keys}")
        
    elif method_name in ['mean', 'sum', 'min', 'max', 'count']:
        # Apply the aggregation function, ensuring it's properly renaming columns
        logger.debug(f"  Applying {method_name} aggregation to schema with columns: {new_schema.columns}")
        
        # Apply aggregation to all non-groupby columns
        if new_schema.in_groupby_chain:
            non_group_cols = [col for col in new_schema.columns if col not in new_schema.group_keys]
            logger.debug(f"  Aggregating non-group columns: {non_group_cols}")
            
            # For group-by aggregations, we need to create column mappings
            for col in non_group_cols:
                agg_col = f"{col}_{method_name}"
                new_schema.aggregated_columns[col] = agg_col
                logger.debug(f"  Creating mapping: {col} -> {agg_col}")
        
        # Apply the aggregation to update the schema state
        new_schema.apply_aggregation(method_name)
        logger.debug(f"  After {method_name} aggregation: columns={new_schema.columns}, agg_cols={new_schema.aggregated_columns}")
        logger.debug(f"  Schema updated for aggregation: {method_name}")
        
    elif method_name == 'sort_values':
        # No schema changes, but log for debugging
        by_arg = None
        if 'by' in kwargs:
            by_arg = kwargs['by']
        elif args:
            by_arg = args[0]
            
        if by_arg:
            if isinstance(by_arg, ast.Constant) and isinstance(by_arg.value, str):
                col_name = by_arg.value
                resolved_col = new_schema.resolve_column_reference(col_name)
                logger.debug(f"  Sort by column: {col_name} -> resolved to: {resolved_col}")
            elif isinstance(by_arg, ast.List):
                col_names = [
                    elt.value for elt in by_arg.elts 
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                resolved_cols = new_schema.resolve_column_references(col_names)
                logger.debug(f"  Sort by columns: {col_names} -> resolved to: {resolved_cols}")
    
    return new_schema


def _extract_groupby_keys(node: ast.Call) -> List[str]:
    """Extract groupby keys from a groupby() call node."""
    keys = []
    
    # Check positional args (first arg is 'by')
    if node.args:
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys = [node.args[0].value]
        elif isinstance(node.args[0], ast.List):
            keys = [
                elt.value for elt in node.args[0].elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    
    # Check keyword args
    for kw in node.keywords:
        if kw.arg == 'by':
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                keys = [kw.value.value]
            elif isinstance(kw.value, ast.List):
                keys = [
                    elt.value for elt in kw.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
            break
    
    return keys 