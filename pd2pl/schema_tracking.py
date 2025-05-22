"""Schema tracking system for method chains.

This module provides classes to track DataFrame schema transformations through
method chains, allowing for accurate column resolution in complex operations.
"""
from typing import Dict, Set, List, Optional, Any, Union
import logging
from pd2pl.logging import logger


class SchemaState:
    """Tracks the evolving schema of a DataFrame through operations.
    
    This class maintains information about:
    - The columns in the current DataFrame
    - The context of operations (groupby, aggregation, etc.)
    - Column origins and transformations
    - Column name mappings for aggregations
    """
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = set(columns or [])  # Current column names
        self.column_origins = {}  # Maps derived columns to source columns
        self.column_aliases = {}  # Maps operation-specific column aliases
        self.group_keys = set()  # Columns used as groupby keys
        self.aggregated_columns = {}  # Maps agg column names to source+operation
        self.in_groupby_chain = False  # Flag for groupby chains
    
    def copy(self) -> 'SchemaState':
        """Create a deep copy of the schema state."""
        new_state = SchemaState()
        new_state.columns = self.columns.copy()
        new_state.column_origins = self.column_origins.copy()
        new_state.column_aliases = self.column_aliases.copy()
        new_state.group_keys = self.group_keys.copy()
        
        # Handle mixed types in aggregated_columns (strings and dicts)
        new_state.aggregated_columns = {}
        for k, v in self.aggregated_columns.items():
            if isinstance(v, dict):
                new_state.aggregated_columns[k] = v.copy()
            else:
                # For string values, just copy the value directly
                new_state.aggregated_columns[k] = v
                
        new_state.in_groupby_chain = self.in_groupby_chain
        return new_state
    
    def apply_selection(self, selected_columns: List[str]) -> 'SchemaState':
        """Update schema based on column selection operation."""
        if selected_columns:
            logger.debug(f"SchemaState.apply_selection: original columns={self.columns}, selected={selected_columns}")
            # If columns is empty, just use the selection directly
            if not self.columns:
                self.columns = set(selected_columns)
            else:
                # Otherwise intersect with existing columns or use selection if intersection is empty
                intersection = set(selected_columns).intersection(self.columns)
                self.columns = intersection if intersection else set(selected_columns)
            logger.debug(f"SchemaState.apply_selection: new columns={self.columns}")
        return self
    
    def apply_groupby(self, group_keys: List[str]) -> 'SchemaState':
        """Update schema for groupby operation."""
        self.in_groupby_chain = True
        logger.debug(f"SchemaState.apply_groupby: original group_keys={self.group_keys}, new keys={group_keys}")
        
        # Make sure all group keys are in columns (implicitly add if missing)
        for key in group_keys:
            if key not in self.columns:
                self.columns.add(key)
                logger.debug(f"SchemaState.apply_groupby: Added missing group key {key} to columns")
        
        # Set group keys
        self.group_keys = set(group_keys)
        logger.debug(f"SchemaState.apply_groupby: updated group_keys={self.group_keys}")
        return self
    
    def apply_aggregation(self, agg_function: str) -> 'SchemaState':
        """Apply aggregation to schema (after groupby)."""
        if not self.in_groupby_chain:
            logger.debug("SchemaState.apply_aggregation: Not in groupby chain, skipping")
            return self
            
        logger.debug(f"SchemaState.apply_aggregation: applying {agg_function} to columns={self.columns}")
        new_columns = set(self.group_keys)  # Group keys remain
        column_mapping = {}  # Track original to renamed columns
        
        # For each non-group column, create an aggregated version
        for col in self.columns:
            if col not in self.group_keys:
                agg_col = f"{col}_{agg_function}"
                new_columns.add(agg_col)
                column_mapping[col] = agg_col
                self.aggregated_columns[col] = agg_col
                self.column_origins[agg_col] = col
                logger.debug(f"SchemaState.apply_aggregation: Mapped {col} -> {agg_col}")
        
        # Update the columns set with the new aggregated column names
        self.columns = new_columns
        logger.debug(f"SchemaState.apply_aggregation: Updated columns={self.columns}")
        logger.debug(f"SchemaState.apply_aggregation: Column mappings={self.aggregated_columns}")
        return self
    
    def apply_aggregation_dict(self, agg_dict: Dict[str, Union[str, List[str]]]) -> 'SchemaState':
        """Apply multiple aggregations specified in a dictionary."""
        if not self.in_groupby_chain:
            logger.debug("SchemaState.apply_aggregation_dict: Not in groupby chain, skipping")
            return self
            
        logger.debug(f"SchemaState.apply_aggregation_dict: applying {agg_dict} to columns={self.columns}")
        new_columns = set(self.group_keys)  # Group keys remain
        
        for col, funcs in agg_dict.items():
            if isinstance(funcs, str):
                funcs = [funcs]
                
            for func in funcs:
                agg_col = f"{col}_{func}"
                self.aggregated_columns[agg_col] = {
                    'source_col': col,
                    'agg_function': func
                }
                new_columns.add(agg_col)
                self.column_origins[agg_col] = col
        
        self.columns = new_columns
        logger.debug(f"SchemaState.apply_aggregation_dict: updated columns={self.columns}")
        logger.debug(f"SchemaState.apply_aggregation_dict: updated aggregated_columns={self.aggregated_columns}")
        return self
    
    def resolve_column_reference(self, col_name: str) -> str:
        """
        Resolve a column reference to its current name in the schema.
        
        This handles cases like referring to 'B' after groupby().mean()
        where the actual column is now 'B_mean'.
        """
        logger.debug(f"SchemaState.resolve_column_reference: resolving {col_name}")
        
        # If column exists directly, return it
        if col_name in self.columns:
            logger.debug(f"SchemaState.resolve_column_reference: {col_name} exists directly")
            return col_name
            
        # Check if it's a source column that's been aggregated
        if self.in_groupby_chain:
            # First check direct mapping in new aggregated_columns structure
            if col_name in self.aggregated_columns:
                resolved = self.aggregated_columns[col_name]
                logger.debug(f"SchemaState.resolve_column_reference: resolved {col_name} -> {resolved}")
                return resolved
                
            # Fallback to legacy structure if needed
            matching_agg_cols = []
            for agg_col, info in self.aggregated_columns.items():
                if isinstance(info, dict) and info.get('source_col') == col_name:
                    matching_agg_cols.append(agg_col)
            
            if matching_agg_cols:
                # If multiple matches, prefer the most recently added one
                resolved = matching_agg_cols[-1]
                logger.debug(f"SchemaState.resolve_column_reference: resolved {col_name} -> {resolved} (legacy)")
                return resolved
                
        # Return original as fallback
        logger.debug(f"SchemaState.resolve_column_reference: no resolution for {col_name}, using as is")
        return col_name

    def resolve_column_references(self, col_names: List[str]) -> List[str]:
        """Resolve multiple column references at once."""
        logger.debug(f"SchemaState.resolve_column_references: resolving {col_names}")
        result = [self.resolve_column_reference(col) for col in col_names]
        logger.debug(f"SchemaState.resolve_column_references: resolved {col_names} -> {result}")
        return result


class SchemaRegistry:
    """Registry for tracking DataFrame schemas throughout translation."""
    def __init__(self):
        self.schemas = {}  # Maps var_name to SchemaState
    
    def register_dataframe(self, var_name: str, columns: Optional[List[str]] = None) -> SchemaState:
        """Register a new DataFrame with optional known columns."""
        logger.debug(f"SchemaRegistry.register_dataframe: {var_name} with columns={columns}")
        self.schemas[var_name] = SchemaState(columns)
        return self.schemas[var_name]
    
    def get_schema(self, var_name: str) -> Optional[SchemaState]:
        """Get schema for a DataFrame variable."""
        result = self.schemas.get(var_name)
        logger.debug(f"SchemaRegistry.get_schema: {var_name} -> {'found' if result else 'not found'}")
        return result
    
    def update_schema(self, var_name: str, schema: SchemaState) -> None:
        """Update schema for a DataFrame variable."""
        logger.debug(f"SchemaRegistry.update_schema: {var_name}")
        self.schemas[var_name] = schema 