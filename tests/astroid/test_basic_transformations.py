"""Tests for basic pandas→polars transformations using TDD approach."""
import ast
import pytest

from pd2pl.astroid_core.chain_transformer import create_transformer, TransformationResult


class TestBasicTransformations:
    """Test basic pandas→polars method transformations."""
    
    def test_simple_head_transformation(self):
        """Test that df.head() transforms to df.head()."""
        # Given: pandas code with head() call
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
result = df.head()
"""
        
        # When: we transform the code
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        # Then: the result should contain polars equivalent
        transformed_code = result.transformed_code
        
        # Should transform pandas import to polars
        assert "import polars as pl" in transformed_code
        assert "pl.DataFrame" in transformed_code
        
        # head() should remain the same (polars has head() too)
        assert "result = df.head()" in transformed_code
        
        # Should track polars imports needed
        assert "polars" in result.polars_imports_needed
    
    def test_simple_tail_transformation(self):
        """Test that df.tail() transforms to df.tail()."""
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
result = df.tail(2)
"""
        
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        transformed_code = result.transformed_code
        
        assert "import polars as pl" in transformed_code
        assert "pl.DataFrame" in transformed_code
        assert "result = df.tail(2)" in transformed_code
        assert "polars" in result.polars_imports_needed
    
    def test_shape_property_transformation(self):
        """Test that df.shape transforms to df.shape."""
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
rows, cols = df.shape
"""
        
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        transformed_code = result.transformed_code
        
        assert "import polars as pl" in transformed_code
        assert "pl.DataFrame" in transformed_code
        # astroid may format tuple assignment differently
        assert "df.shape" in transformed_code
        assert "rows" in transformed_code and "cols" in transformed_code
    
    def test_no_transformation_when_no_pandas(self):
        """Test that non-pandas code is left unchanged."""
        non_pandas_code = """
x = [1, 2, 3]
result = len(x)
"""
        
        transformer = create_transformer(non_pandas_code)
        result = transformer.transform_chains(non_pandas_code)
        
        transformed_code = result.transformed_code
        
        # Should be unchanged
        assert transformed_code.strip() == non_pandas_code.strip()
        assert result.polars_imports_needed == []
    
    def test_transformation_warnings(self):
        """Test that transformations can generate warnings."""
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
result = df.head()
"""
        
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        # Should have a TransformationResult with warnings list
        assert isinstance(result.warnings, list)
        # Warnings might be empty for simple transformations, but structure should exist 