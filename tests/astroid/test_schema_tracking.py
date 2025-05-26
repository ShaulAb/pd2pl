"""
Tests for the enhanced AstroidSchemaState with column type tracking.

This module tests the column type inference capabilities and schema tracking of
the astroid-based schema system.
"""

import pytest
import astroid
from astroid import nodes

from pd2pl.astroid_core.schema_tracking import AstroidSchemaState, AstroidSchemaRegistry


def test_column_types_in_constructor():
    """Test column type inference from DataFrame constructor."""
    # Create a sample DataFrame constructor node
    code = """
    import pandas as pd
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })
    """
    
    module = astroid.parse(code)
    # Find the assignment node directly
    for node in module.body:
        if isinstance(node, nodes.Assign):
            assign_node = node
            break
    else:
        pytest.fail("Could not find assignment node in the AST")
    
    # Create schema and infer columns
    schema = AstroidSchemaState(name="df")
    schema.infer_columns_from_node(assign_node.value)
    
    # Check column inference
    assert 'int_col' in schema.columns
    assert 'float_col' in schema.columns
    assert 'str_col' in schema.columns
    assert 'bool_col' in schema.columns
    
    # Check type inference
    assert schema.get_column_type('int_col') == 'int'
    assert schema.get_column_type('float_col') == 'float'
    assert schema.get_column_type('str_col') == 'str'
    assert schema.get_column_type('bool_col') == 'bool'


def test_column_types_with_dtype():
    """Test column type inference from DataFrame with explicit dtype."""
    # Create a sample DataFrame with dtype specification
    code = """
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [1.1, 2.2, 3.3]
    }, dtype={'col1': 'int32', 'col2': 'float64'})
    """
    
    module = astroid.parse(code)
    # Find the assignment node directly
    for node in module.body:
        if isinstance(node, nodes.Assign):
            assign_node = node
            break
    else:
        pytest.fail("Could not find assignment node in the AST")
    
    # Create schema and infer columns
    schema = AstroidSchemaState(name="df")
    schema.infer_columns_from_node(assign_node.value)
    
    # Check column inference
    assert 'col1' in schema.columns
    assert 'col2' in schema.columns
    
    # Check type inference (should normalize types)
    assert schema.get_column_type('col1') == 'int'
    assert schema.get_column_type('col2') == 'float'


def test_column_types_with_single_dtype():
    """Test column type inference with a single dtype for all columns."""
    code = """
    import pandas as pd
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    }, dtype='float64')
    """
    
    module = astroid.parse(code)
    # Find the assignment node directly
    for node in module.body:
        if isinstance(node, nodes.Assign):
            assign_node = node
            break
    else:
        pytest.fail("Could not find assignment node in the AST")
    
    # Create schema and infer columns
    schema = AstroidSchemaState(name="df")
    schema.infer_columns_from_node(assign_node.value)
    
    # Both columns should have float type despite containing integers
    assert schema.get_column_type('col1') == 'float'
    assert schema.get_column_type('col2') == 'float'


def test_aggregation_type_inference():
    """Test type inference for aggregated columns."""
    schema = AstroidSchemaState(name="df")
    
    # Add some columns with known types
    schema.columns = {'A', 'B', 'C', 'D'}
    schema.column_types = {
        'A': 'int',
        'B': 'float', 
        'C': 'str',
        'D': 'bool'
    }
    
    # Apply a groupby
    schema.apply_groupby(['A'])
    
    # Apply different aggregations
    schema.apply_aggregation('mean')
    
    # Check the inferred types for aggregated columns
    assert schema.get_column_type('B_mean') == 'float64'
    assert schema.get_column_type('D_mean') == 'float64'  # bool.mean() -> float64
    assert 'C_mean' in schema.columns  # str columns get aggregated too, even if not meaningful
    
    # Their types should be preserved or converted appropriately
    assert schema.get_column_type('A') == 'int'  # Group key preserved


def test_schema_registry_with_types():
    """Test schema propagation in the registry, including types."""
    registry = AstroidSchemaRegistry()
    
    # Register a dataframe with initial columns and types
    registry.register_dataframe("df1", 
                              columns=["A", "B", "C"],
                              column_types={"A": "int", "B": "float", "C": "str"})
    
    # Copy schema to a new dataframe
    registry.copy_schema("df1", "df2")
    
    # Get and verify the copied schema
    df2_schema = registry.get_schema("df2")
    assert df2_schema is not None
    assert df2_schema.name == "df2"
    assert "A" in df2_schema.columns
    assert "B" in df2_schema.columns
    assert "C" in df2_schema.columns
    
    # Verify types were copied
    assert df2_schema.get_column_type("A") == "int"
    assert df2_schema.get_column_type("B") == "float"
    assert df2_schema.get_column_type("C") == "str"
    
    # Modify the second schema
    df2_schema.columns.add("D")
    df2_schema.column_types["D"] = "datetime"
    
    # Original schema should be unchanged
    df1_schema = registry.get_schema("df1")
    assert "D" not in df1_schema.columns
    assert "D" not in df1_schema.column_types


def test_merge_inferred_types():
    """Test merging of multiple inferred types."""
    schema = AstroidSchemaState(name="df")
    
    # Test single type
    assert schema._merge_inferred_types({"int"}) == "int"
    
    # Test numeric type hierarchy
    assert schema._merge_inferred_types({"int", "float"}) == "float"
    assert schema._merge_inferred_types({"int", "float", "complex"}) == "complex"
    
    # Test mixed types
    # Updated to match our implementation
    assert schema._merge_inferred_types({"int", "str"}) == "object"
    
    # Test empty set
    assert schema._merge_inferred_types(set()) == "unknown"


def test_normalize_dtype_str():
    """Test normalization of dtype strings."""
    schema = AstroidSchemaState(name="df")
    
    # Test various int types
    assert schema._normalize_dtype_str("int64") == "int"
    assert schema._normalize_dtype_str("Int32") == "int"
    assert schema._normalize_dtype_str("integer") == "int"
    
    # Test float types
    assert schema._normalize_dtype_str("float32") == "float"
    assert schema._normalize_dtype_str("Float64") == "float"
    assert schema._normalize_dtype_str("double") == "float"
    
    # Test other types
    assert schema._normalize_dtype_str("object") == "str"
    assert schema._normalize_dtype_str("category") == "category"
    assert schema._normalize_dtype_str("datetime64") == "datetime"
    
    # Test non-string value
    assert schema._normalize_dtype_str(None) == "unknown" 