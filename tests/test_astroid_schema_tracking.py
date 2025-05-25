"""
Tests for the astroid-based schema tracking functionality.

These tests verify that the AstroidSchemaState and AstroidSchemaRegistry
classes correctly track DataFrame schemas using astroid's type inference
and parent-child relationships.
"""

import pytest
import astroid
from pd2pl.astroid_core.schema_tracking import AstroidSchemaState, AstroidSchemaRegistry

def test_basic_schema_state():
    """Test basic schema state creation and manipulation."""
    # Create a schema state
    schema = AstroidSchemaState(name='df')
    
    # Add some columns
    schema.columns.update(['a', 'b', 'c'])
    
    # Verify columns were added
    assert 'a' in schema.columns
    assert 'b' in schema.columns
    assert 'c' in schema.columns
    assert len(schema.columns) == 3

def test_groupby_aggregation():
    """Test schema transformation through groupby and aggregation."""
    # Create a schema state
    schema = AstroidSchemaState(name='df')
    schema.columns.update(['category', 'value1', 'value2'])
    
    # Apply groupby
    schema.apply_groupby(['category'])
    
    # Verify groupby state
    assert schema.in_groupby_chain
    assert 'category' in schema.group_keys
    assert len(schema.group_keys) == 1
    
    # Apply aggregation
    schema.apply_aggregation('mean')
    
    # Verify column transformations
    assert 'category' in schema.columns
    assert 'value1_mean' in schema.columns
    assert 'value2_mean' in schema.columns
    assert schema.aggregated_columns['value1'] == 'value1_mean'
    assert schema.aggregated_columns['value2'] == 'value2_mean'
    assert schema.column_origins['value1_mean'] == 'value1'
    assert schema.column_origins['value2_mean'] == 'value2'
    
    # Test pandas compat rename map
    rename_map = schema.get_pandas_compat_rename_map()
    assert rename_map['value1_mean'] == 'value1'
    assert rename_map['value2_mean'] == 'value2'

def test_column_inference_from_dataframe_constructor():
    """Test inferring columns from a DataFrame constructor."""
    # Create a schema registry
    registry = AstroidSchemaRegistry()
    
    # Parse a simple DataFrame creation
    code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
"""
    module = astroid.parse(code)
    
    # Find the DataFrame constructor node
    df_node = None
    for node in module.body:
        if isinstance(node, astroid.Assign):
            if node.targets[0].name == 'df':
                df_node = node.value
                break
    
    assert df_node is not None, "Could not find DataFrame constructor"
    
    # Create a schema
    schema = registry.register_dataframe('df')
    
    # Infer columns from the constructor
    schema.infer_columns_from_node(df_node)
    
    # Verify inferred columns
    assert 'a' in schema.columns
    assert 'b' in schema.columns
    assert len(schema.columns) == 2

def test_column_inference_from_subscript():
    """Test inferring columns from subscript operations."""
    # Create a schema registry
    registry = AstroidSchemaRegistry()
    
    # Parse code with subscript operations
    code = """
import pandas as pd
df = pd.DataFrame()
x = df['column1']
y = df[['column2', 'column3']]
"""
    module = astroid.parse(code)
    
    # Find the subscript nodes
    subscript_nodes = []
    for node in module.body:
        if isinstance(node, astroid.Assign):
            if isinstance(node.value, astroid.Subscript):
                subscript_nodes.append(node.value)
    
    assert len(subscript_nodes) == 2, "Could not find subscript operations"
    
    # Create a schema
    schema = registry.register_dataframe('df')
    
    # Infer columns from subscripts
    for node in subscript_nodes:
        schema.infer_columns_from_node(node)
    
    # Verify inferred columns
    assert 'column1' in schema.columns
    assert 'column2' in schema.columns
    assert 'column3' in schema.columns
    assert len(schema.columns) == 3

def test_parent_child_relationship():
    """Test parent-child relationship tracking for context discovery."""
    # Parse a chain of operations
    code = """
import pandas as pd
result = df.groupby('category').mean().sort_values(by='value1')
"""
    module = astroid.parse(code)
    
    # Find the method chain node
    chain_node = None
    for node in module.body:
        if isinstance(node, astroid.Assign):
            if node.targets[0].name == 'result':
                chain_node = node.value
                break
    
    assert chain_node is not None, "Could not find method chain"
    
    # Verify parent-child relationships
    # sort_values node
    assert isinstance(chain_node, astroid.Call)
    assert chain_node.func.attrname == 'sort_values'
    
    # mean node (parent of sort_values)
    mean_node = chain_node.func.expr
    assert isinstance(mean_node, astroid.Call)
    assert mean_node.func.attrname == 'mean'
    
    # groupby node (parent of mean)
    groupby_node = mean_node.func.expr
    assert isinstance(groupby_node, astroid.Call)
    assert groupby_node.func.attrname == 'groupby'
    
    # df node (parent of groupby)
    df_node = groupby_node.func.expr
    assert isinstance(df_node, astroid.Name)
    assert df_node.name == 'df'
    
    # Test traversing up the chain
    current = chain_node
    
    # Get the sort_values 'by' parameter
    by_param = None
    for keyword in current.keywords:
        if keyword.arg == 'by':
            by_param = keyword.value.value
            break
    
    assert by_param == 'value1'
    
    # Move up to mean node
    parent = current.func.expr
    assert parent.func.attrname == 'mean'
    
    # Move up to groupby node
    grandparent = parent.func.expr
    assert grandparent.func.attrname == 'groupby'
    
    # Get the groupby key
    group_key = grandparent.args[0].value
    assert group_key == 'category'

def test_context_discovery_in_method_chain():
    """Test discovering context from method chains."""
    # Create a schema registry
    registry = AstroidSchemaRegistry()
    
    # Parse a chain of operations
    code = """
import pandas as pd
df = pd.DataFrame({'category': ['A', 'B', 'A'], 'value1': [1, 2, 3], 'value2': [4, 5, 6]})
result = df.groupby('category').mean().sort_values(by='value1')
"""
    module = astroid.parse(code)
    
    # Find the method chain node
    chain_node = None
    for node in module.body:
        if isinstance(node, astroid.Assign):
            if node.targets[0].name == 'result':
                chain_node = node.value
                break
    
    assert chain_node is not None, "Could not find method chain"
    
    # Register the DataFrame
    schema = registry.register_dataframe('df')
    
    # Simulate discovering columns from the DataFrame constructor
    df_constructor = None
    for node in module.body:
        if isinstance(node, astroid.Assign):
            if node.targets[0].name == 'df':
                df_constructor = node.value
                break
    
    schema.infer_columns_from_node(df_constructor)
    
    # Verify columns were correctly inferred
    assert 'category' in schema.columns
    assert 'value1' in schema.columns
    assert 'value2' in schema.columns
    
    # Now let's process the method chain
    # First, extract groupby key from the chain
    group_key = chain_node.func.expr.func.expr.args[0].value
    schema.apply_groupby([group_key])
    
    # Apply the aggregation
    schema.apply_aggregation('mean')
    
    # Verify column transformations
    assert 'category' in schema.columns
    assert 'value1_mean' in schema.columns
    assert 'value2_mean' in schema.columns
    
    # Verify rename map
    rename_map = schema.get_pandas_compat_rename_map()
    assert rename_map['value1_mean'] == 'value1'
    assert rename_map['value2_mean'] == 'value2'
