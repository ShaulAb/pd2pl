"""Tests for the new chain transformation engine."""
import ast
import pytest

from pd2pl.astroid_core.chain_transformer import ChainTransformer, TransformationResult, create_transformer
from pd2pl.astroid_core.chain_discovery import ChainRegistry, MethodChain
from pd2pl.astroid_core.schema_tracking import AstroidSchemaRegistry


class TestChainTransformer:
    """Test the new clean transformation engine."""
    
    def test_transformer_initialization(self):
        """Test that we can initialize the transformer with registries."""
        chain_registry = ChainRegistry()
        schema_registry = AstroidSchemaRegistry()
        
        transformer = ChainTransformer(chain_registry, schema_registry)
        
        assert transformer.chain_registry is chain_registry
        assert transformer.schema_registry is schema_registry
        assert transformer.polars_imports_needed == set()
        assert transformer.warnings == []
    
    def test_create_transformer_factory(self):
        """Test the factory function that runs discovery and creates transformer."""
        # Simple pandas code with a basic chain
        code = """
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.groupby('A').mean()
"""
        
        transformer = create_transformer(code)
        
        assert isinstance(transformer, ChainTransformer)
        assert isinstance(transformer.chain_registry, ChainRegistry)
        assert isinstance(transformer.schema_registry, AstroidSchemaRegistry)
    
    def test_transform_chains_basic_interface(self):
        """Test that transform_chains returns proper result structure."""
        chain_registry = ChainRegistry()
        schema_registry = AstroidSchemaRegistry()
        transformer = ChainTransformer(chain_registry, schema_registry)
        
        # Simple code string
        code = "df.head()"
        
        result = transformer.transform_chains(code)
        
        assert isinstance(result, TransformationResult)
        assert isinstance(result.transformed_code, str)
        assert isinstance(result.polars_imports_needed, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.schema_changes, dict)
    
    def test_transform_chains_preserves_code_when_no_chains(self):
        """Test that code is preserved when no chains are discovered."""
        chain_registry = ChainRegistry()  # Empty registry
        schema_registry = AstroidSchemaRegistry()
        transformer = ChainTransformer(chain_registry, schema_registry)
        
        code = "x = 1 + 2"  # No pandas code
        
        result = transformer.transform_chains(code)
        
        # Should return the same code structure
        assert result.transformed_code.strip() == code.strip()
        assert result.polars_imports_needed == []
        assert result.warnings == []
    
    def test_integration_with_discovery_system(self):
        """Test that transformer integrates properly with our discovery system."""
        # Code with a simple chain that our discovery system should find
        code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.groupby('A').mean()
"""
        
        transformer = create_transformer(code)
        # Use the pure astroid interface
        result = transformer.transform_chains(code)
        
        # Should have discovered the chain
        chains = transformer.chain_registry.get_all_chains()
        assert len(chains) > 0, "Should have discovered at least one chain"
        
        # Should have a valid transformation result
        assert isinstance(result, TransformationResult)
        assert isinstance(result.transformed_code, str) 