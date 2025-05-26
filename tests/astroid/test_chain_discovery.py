import pytest
from pd2pl.astroid_core.chain_discovery import ChainDiscovery
import astroid

@pytest.mark.parametrize(
    "pandas_code,expected_methods",
    [
        ("df.filter(items=['A']).rename(columns={'A': 'C'})", ["filter", "rename"]),
        ("df[df.A > 1].sort_values('B')", ["__getitem__", "sort_values"]),
    ]
)
def test_simple_chain_discovery(pandas_code, expected_methods):
    # Parse code to astroid module
    code = f"result = {pandas_code}"
    module = astroid.parse(code)
    # Discover chains
    chains = ChainDiscovery().discover_chains(module)
    # There should be exactly one chain
    assert len(chains) == 1
    chain = chains[0]
    # Extract method names from the chain
    method_names = [m.name for m in chain.methods]
    assert method_names == expected_methods


@pytest.mark.parametrize(
    "pandas_code,expected_methods",
    [
        # Three-method chain
        ("df.filter(items=['A']).rename(columns={'A': 'C'}).sort_values('C')", 
         ["filter", "rename", "sort_values"]),
        
        # Complex boolean indexing with method chain
        ("df[(df.A > 1) & (df.B < 10)].groupby('category').mean()", 
         ["__getitem__", "groupby", "mean"]),
        
        # Chain with groupby and aggregation
        ("df.groupby('category').agg({'values': 'mean'}).reset_index()", 
         ["groupby", "agg", "reset_index"]),
        
        # Chain with multiple filters
        ("df[df.A > 1][df.B < 10].sort_values('C')", 
         ["__getitem__", "__getitem__", "sort_values"]),
        
        # Long chain with mixed operations
        ("df.dropna().filter(items=['A', 'B']).rename(columns={'A': 'X'}).sort_values('B').head(10)", 
         ["dropna", "filter", "rename", "sort_values", "head"]),
    ]
)
def test_complex_chain_discovery(pandas_code, expected_methods):
    """Test discovery of more complex method chains."""
    # Parse code to astroid module
    code = f"result = {pandas_code}"
    module = astroid.parse(code)
    # Discover chains
    chains = ChainDiscovery().discover_chains(module)
    # There should be exactly one chain
    assert len(chains) == 1
    chain = chains[0]
    # Extract method names from the chain
    method_names = [m.name for m in chain.methods]
    assert method_names == expected_methods


def test_multiple_chains_in_module():
    """Test discovery of multiple chains in the same module."""
    code = """
result1 = df.filter(items=['A']).rename(columns={'A': 'C'})
result2 = df[df.B > 5].sort_values('B')
result3 = df.groupby('category').mean()
"""
    module = astroid.parse(code)
    chains = ChainDiscovery().discover_chains(module)
    
    # Should find three chains
    assert len(chains) == 3
    
    # Check each chain
    expected_chains = [
        ["filter", "rename"],
        ["__getitem__", "sort_values"],
        ["groupby", "mean"]
    ]
    
    actual_chains = [[m.name for m in chain.methods] for chain in chains]
    assert actual_chains == expected_chains


def test_no_chains_found():
    """Test cases where no chains should be found."""
    test_cases = [
        "result = df",  # Simple variable assignment
        "result = 42",  # Non-DataFrame assignment
        "print('hello')",  # Function call, not assignment
        "df.shape",  # Property access, not method chain
    ]
    
    for code in test_cases:
        module = astroid.parse(code)
        chains = ChainDiscovery().discover_chains(module)
        assert len(chains) == 0, f"Expected no chains for: {code}"


def test_chain_metadata():
    """Test that chain metadata is correctly populated."""
    code = "result = df.filter(items=['A']).rename(columns={'A': 'C'})"
    module = astroid.parse(code)
    chains = ChainDiscovery().discover_chains(module)
    
    assert len(chains) == 1
    chain = chains[0]
    
    # Check that start and end nodes are set
    assert chain.start_node is not None
    assert chain.end_node is not None
    
    # Check that methods have correct structure
    assert len(chain.methods) == 2
    
    filter_method = chain.methods[0]
    assert filter_method.name == "filter"
    assert filter_method.node is not None
    assert isinstance(filter_method.args, list)
    assert isinstance(filter_method.kwargs, dict)
    
    rename_method = chain.methods[1]
    assert rename_method.name == "rename"
    assert rename_method.node is not None 