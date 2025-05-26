"""
Tests for schema integration in chain discovery.

This module tests the integration between chain discovery and schema tracking,
ensuring that schema changes are properly tracked through method chains.
"""

import pytest
import astroid
from pd2pl.astroid_core.chain_discovery import ChainDiscovery, ChainRegistry, ChainType
from pd2pl.astroid_core.schema_tracking import AstroidSchemaRegistry, AstroidSchemaState


class TestChainSchemaIntegration:
    """Test schema integration with chain discovery."""
    
    def test_basic_schema_tracking(self):
        """Test basic schema tracking through a simple chain."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.groupby('A').mean()
"""
        module = astroid.parse(code)
        
        # Create schema registry and register initial DataFrame
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B'], {'A': 'int', 'B': 'int'})
        
        # Discover chains with schema tracking
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Check schema tracking
        assert chain.input_schema is not None
        assert chain.output_schema is not None
        assert chain.variable_name == 'result'
        assert len(chain.schema_changes) == 2  # groupby + mean
        
        # Check input schema
        assert 'A' in chain.input_schema.columns
        assert 'B' in chain.input_schema.columns
        
        # Check output schema (after groupby + mean)
        assert 'A' in chain.output_schema.columns  # Group key remains
        assert 'B_mean' in chain.output_schema.columns  # Aggregated column
        assert 'B' not in chain.output_schema.columns  # Original column replaced
    
    def test_column_selection_schema_tracking(self):
        """Test schema tracking for column selection."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
result = df[['A', 'B']]
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B', 'C'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Check schema changes
        assert len(chain.schema_changes) == 1
        assert "Selected columns: ['A', 'B']" in chain.schema_changes[0]
        
        # Check output schema
        assert 'A' in chain.output_schema.columns
        assert 'B' in chain.output_schema.columns
        assert 'C' not in chain.output_schema.columns
    
    def test_rename_schema_tracking(self):
        """Test schema tracking for column renaming."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.rename(columns={'A': 'X', 'B': 'Y'})
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Check schema changes
        assert len(chain.schema_changes) == 1
        assert "Renamed columns:" in chain.schema_changes[0]
        
        # Check output schema
        assert 'X' in chain.output_schema.columns
        assert 'Y' in chain.output_schema.columns
        assert 'A' not in chain.output_schema.columns
        assert 'B' not in chain.output_schema.columns
    
    def test_complex_chain_schema_tracking(self):
        """Test schema tracking through a complex chain."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
result = df[df.A > 1].groupby('B').agg({'C': 'sum'}).reset_index()
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B', 'C'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Check that we have multiple schema changes
        assert len(chain.schema_changes) >= 3  # boolean indexing, groupby, agg, reset_index
        
        # Check final schema includes group key and aggregated column
        assert 'B' in chain.output_schema.columns
        assert 'C_sum' in chain.output_schema.columns
    
    def test_schema_registry_integration(self):
        """Test that schema registry is properly updated."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = df.groupby('A').mean()
df3 = df2.rename(columns={'B_mean': 'B_average'})
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 2
        
        # Check that df2 schema was registered
        df2_schema = schema_registry.get_schema('df2')
        assert df2_schema is not None
        assert 'A' in df2_schema.columns
        assert 'B_mean' in df2_schema.columns
        
        # Check that df3 schema was registered
        df3_schema = schema_registry.get_schema('df3')
        assert df3_schema is not None
        assert 'A' in df3_schema.columns
        assert 'B_average' in df3_schema.columns
        assert 'B_mean' not in df3_schema.columns
    
    def test_chain_registry_schema_queries(self):
        """Test schema-based queries on chain registry."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
result1 = df.groupby('A').mean()
result2 = df[['A', 'B']]
result3 = df.rename(columns={'A': 'X'})
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B', 'C'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 3
        
        # Test schema change queries
        chains_with_changes = discovery.registry.get_chains_with_schema_changes()
        assert len(chains_with_changes) == 3
        
        # Test specific schema change types
        groupby_chains = discovery.registry.get_chains_by_schema_change_type('groupby')
        assert len(groupby_chains) == 1
        
        rename_chains = discovery.registry.get_chains_by_schema_change_type('renamed')
        assert len(rename_chains) == 1
        
        selection_chains = discovery.registry.get_chains_by_schema_change_type('selected')
        assert len(selection_chains) == 1
    
    def test_schema_evolution_tracking(self):
        """Test tracking schema evolution for a DataFrame."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df = df.rename(columns={'A': 'X'})
df = df.groupby('X').mean()
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        # Get schema evolution
        evolution = discovery.registry.get_schema_evolution('df')
        
        # Should have multiple schema states showing evolution
        assert len(evolution) >= 2
        
        # First state should have original columns
        first_state = evolution[0]
        assert 'A' in first_state.columns
        assert 'B' in first_state.columns
    
    def test_no_schema_changes_for_simple_operations(self):
        """Test that simple operations without schema changes are handled correctly."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.reset_index()
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Should have one schema change (reset_index note)
        assert len(chain.schema_changes) == 1
        assert "Reset index" in chain.schema_changes[0]
        
        # Schema should remain the same
        assert chain.input_schema.columns == chain.output_schema.columns
    
    def test_boolean_indexing_schema_tracking(self):
        """Test schema tracking for boolean indexing operations."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df[df.A > 1]
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Should have one schema change for boolean indexing
        assert len(chain.schema_changes) == 1
        assert "boolean indexing" in chain.schema_changes[0]
        
        # Columns should remain the same (boolean indexing doesn't change schema)
        assert chain.input_schema.columns == chain.output_schema.columns
    
    def test_drop_columns_schema_tracking(self):
        """Test schema tracking for column dropping."""
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
result = df.drop(columns=['C'])
"""
        module = astroid.parse(code)
        
        schema_registry = AstroidSchemaRegistry()
        schema_registry.register_dataframe('df', ['A', 'B', 'C'])
        
        discovery = ChainDiscovery(schema_registry=schema_registry)
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        chain = chains[0]
        
        # Should have one schema change for dropping columns
        assert len(chain.schema_changes) == 1
        assert "Dropped columns: ['C']" in chain.schema_changes[0]
        
        # Output schema should not have column C
        assert 'A' in chain.output_schema.columns
        assert 'B' in chain.output_schema.columns
        assert 'C' not in chain.output_schema.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 