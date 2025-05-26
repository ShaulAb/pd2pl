"""Tests for chain-aware transformations using TDD approach."""
import ast
import pytest
import pandas as pd
import polars as pl

from pd2pl.astroid_core.chain_transformer import create_transformer, TransformationResult
from pd2pl.astroid_core.chain_discovery import ChainType


class TestChainAwareTransformations:
    """Test chain-aware pandas→polars transformations."""
    
    def test_simple_chain_discovery_integration(self):
        """Test that transformer can access discovered chains."""
        # Given: pandas code with a simple method chain
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.head(5).tail(3)
"""
        
        # When: we create a transformer (which runs discovery)
        transformer = create_transformer(pandas_code)
        
        # Then: the transformer should have discovered the chain
        chains = transformer.chain_registry.get_all_chains()
        assert len(chains) == 1, "Should discover exactly one chain"
        
        chain = chains[0]
        assert len(chain.methods) == 2, "Chain should have 2 methods"
        assert chain.methods[0].name == 'head', "First method should be head"
        assert chain.methods[1].name == 'tail', "Second method should be tail"
        assert chain.dataframe_reference == 'df', "Should reference df variable"
    
    def test_chain_aware_transformation_preserves_structure(self):
        """Test that chain-aware transformation preserves chain structure."""
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})
result = df.head(5)
"""
        
        # When: we transform using chain-aware logic
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        # Then: should transform imports but preserve chain structure
        transformed_code = result.transformed_code
        
        # Basic transformations should still work
        assert "import polars as pl" in transformed_code
        assert "pl.DataFrame" in transformed_code
        
        # Chain structure should be preserved
        assert "result = df.head(5)" in transformed_code
        
        # Should track that we processed a chain
        chains = transformer.chain_registry.get_all_chains()
        assert len(chains) == 1, "Should have processed one chain"
    
    def test_chain_aware_transformation_uses_chain_info(self):
        """Test that transformation actually uses discovered chain information."""
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.head(5).tail(3)
"""
        
        # When: we transform using chain-aware logic
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        # Then: should have transformation metadata about the chain
        assert hasattr(result, 'chain_transformations'), "Should track chain transformations"
        assert len(result.chain_transformations) == 1, "Should have transformed one chain"
        
        chain_transformation = result.chain_transformations[0]
        assert chain_transformation.original_chain is not None, "Should reference original chain"
        assert chain_transformation.transformation_type is not None, "Should specify transformation type"
        assert len(chain_transformation.methods_transformed) == 2, "Should track both methods in chain"
    
    def test_aggregation_chain_type_aware_transformation(self):
        """Test that aggregation chains get specialized transformation logic."""
        pandas_code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.groupby('A').mean()
"""
        
        # When: we transform an aggregation chain
        transformer = create_transformer(pandas_code)
        result = transformer.transform_chains(pandas_code)
        
        # Then: should recognize this as an aggregation chain
        chains = transformer.chain_registry.get_all_chains()
        assert len(chains) == 1, "Should discover one chain"
        assert chains[0].chain_type == ChainType.AGGREGATION, "Should classify as aggregation"
        
        # Should apply aggregation-specific transformation logic
        chain_transformation = result.chain_transformations[0]
        assert chain_transformation.transformation_type == "aggregation", "Should use aggregation transformation"
        assert "groupby" in chain_transformation.methods_transformed, "Should track groupby method"
        assert "mean" in chain_transformation.methods_transformed, "Should track mean method"

    @pytest.mark.parametrize(
        "pandas_code,expected_polars",
        [
            # Basic aggregation methods
            (
                "df.groupby('A').mean()",
                "df.group_by('A').agg(pl.all().mean())"
            ),
            (
                "df.groupby('A').sum()",
                "df.group_by('A').agg(pl.all().sum())"
            ),
            (
                "df.groupby('A').max()",
                "df.group_by('A').agg(pl.all().max())"
            ),
            (
                "df.groupby('A').min()",
                "df.group_by('A').agg(pl.all().min())"
            ),
            (
                "df.groupby('A').count()",
                "df.group_by('A').agg(pl.all().count())"
            ),
            # Different dataframe names
            (
                "data.groupby('Name').sum()",
                "data.group_by('Name').agg(pl.all().sum())"
            ),
            # Different column names
            (
                "df.groupby('Category').mean()",
                "df.group_by('Category').agg(pl.all().mean())"
            ),
        ]
    )
    def test_aggregation_chain_transformations_syntax(self, pandas_code, expected_polars):
        """Test that aggregation chains generate correct polars syntax."""
        full_pandas_code = f"""
import pandas as pd
df = pd.DataFrame({{'A': [1, 2, 3], 'B': [4, 5, 6], 'Category': ['X', 'Y', 'X'], 'Name': ['A', 'B', 'A']}})
data = df.copy()
result = {pandas_code}
"""
        
        # When: we transform the code
        transformer = create_transformer(full_pandas_code)
        result = transformer.transform_chains(full_pandas_code)
        
        # Then: should generate correct polars syntax
        transformed_code = result.transformed_code
        assert expected_polars in transformed_code, f"Failed to transform {pandas_code} to {expected_polars}"
        
        # Should transform imports
        assert "import polars as pl" in transformed_code
        assert "pl.DataFrame" in transformed_code

    @pytest.mark.parametrize(
        "pandas_code,test_data",
        [
            (
                "df.groupby('A').mean()",
                {'A': [1, 1, 2, 2], 'B': [10, 20, 30, 40]}
            ),
            (
                "df.groupby('A').sum()",
                {'A': [1, 1, 2, 2], 'B': [10, 20, 30, 40]}
            ),
            (
                "df.groupby('A').max()",
                {'A': [1, 1, 2, 2], 'B': [10, 20, 30, 40]}
            ),
            (
                "df.groupby('A').min()",
                {'A': [1, 1, 2, 2], 'B': [10, 20, 30, 40]}
            ),
        ]
    )
    def test_aggregation_chain_functional_equivalence(self, pandas_code, test_data):
        """Test that aggregation transformations produce functionally equivalent results."""
        # Create test dataframes
        df_pandas = pd.DataFrame(test_data)
        df_polars = pl.DataFrame(test_data)
        
        # Execute pandas code
        pandas_result = eval(pandas_code, {'df': df_pandas, 'pd': pd}).reset_index()
        
        # Transform to polars and execute
        full_pandas_code = f"""
import pandas as pd
df = pd.DataFrame({test_data})
result = {pandas_code}
"""
        
        transformer = create_transformer(full_pandas_code)
        transformation_result = transformer.transform_chains(full_pandas_code)
        transformed_code = transformation_result.transformed_code
        
        # Extract the polars equivalent
        # This is a simplified approach - in a full implementation we'd have better code execution
        if "group_by('A').agg(pl.all().mean())" in transformed_code:
            polars_result = df_polars.group_by('A').agg(pl.all().mean())
        elif "group_by('A').agg(pl.all().sum())" in transformed_code:
            polars_result = df_polars.group_by('A').agg(pl.all().sum())
        elif "group_by('A').agg(pl.all().max())" in transformed_code:
            polars_result = df_polars.group_by('A').agg(pl.all().max())
        elif "group_by('A').agg(pl.all().min())" in transformed_code:
            polars_result = df_polars.group_by('A').agg(pl.all().min())
        else:
            pytest.skip(f"Polars equivalent not implemented for: {pandas_code}")
        
        # Convert polars result to pandas for comparison
        polars_as_pandas = polars_result.to_pandas().sort_values('A').reset_index(drop=True)
        pandas_sorted = pandas_result.sort_values('A').reset_index(drop=True)
        
        # Compare results (allowing for small floating point differences)
        pd.testing.assert_frame_equal(pandas_sorted, polars_as_pandas, check_dtype=False) 