import pytest
import astroid
from pd2pl.astroid_core.chain_discovery import ChainDiscovery, ChainRegistry, ChainType


def test_chain_classification():
    """Test that chains are correctly classified by type."""
    test_cases = [
        # Filter chains
        ("df[df.A > 1]", ChainType.FILTER),
        ("df.filter(items=['A'])", ChainType.FILTER),
        
        # Transform chains
        ("df.rename(columns={'A': 'B'})", ChainType.TRANSFORM),
        ("df.sort_values('A')", ChainType.TRANSFORM),
        
        # Aggregation chains
        ("df.groupby('A').mean()", ChainType.AGGREGATION),
        ("df.agg({'A': 'sum'})", ChainType.AGGREGATION),
        
        # Mixed chains
        ("df[df.A > 1].groupby('B').mean()", ChainType.MIXED),
        ("df.filter(items=['A']).rename(columns={'A': 'B'})", ChainType.MIXED),
    ]
    
    for pandas_code, expected_type in test_cases:
        code = f"result = {pandas_code}"
        module = astroid.parse(code)
        discovery = ChainDiscovery()
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        assert chains[0].chain_type == expected_type


def test_complexity_scoring():
    """Test that complexity scores are calculated correctly."""
    test_cases = [
        # Simple chains (base score = number of methods)
        ("df.rename(columns={'A': 'B'})", 1.0),
        ("df.filter(items=['A']).rename(columns={'A': 'B'})", 2.0),
        
        # Boolean indexing adds 0.5
        ("df[df.A > 1]", 1.5),
        
        # Groupby adds 1.0
        ("df.groupby('A').mean()", 3.0),  # 2 methods + 1.0 for groupby
        
        # Complex aggregation adds 0.5
        ("df.groupby('A').agg({'B': 'mean'})", 3.5),  # 2 methods + 1.0 for groupby + 0.5 for agg with kwargs
        
        # Combined complexity
        ("df[df.A > 1].groupby('B').agg({'C': 'sum'})", 5.0),  # 3 methods + 0.5 for indexing + 1.0 for groupby + 0.5 for agg
    ]
    
    for pandas_code, expected_score in test_cases:
        code = f"result = {pandas_code}"
        module = astroid.parse(code)
        discovery = ChainDiscovery()
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        assert chains[0].complexity_score == expected_score


def test_dataframe_reference_extraction():
    """Test that DataFrame references are correctly extracted."""
    test_cases = [
        ("df.filter(items=['A'])", "df"),
        ("my_dataframe.groupby('A').mean()", "my_dataframe"),
        ("data[data.A > 1].sort_values('B')", "data"),
    ]
    
    for pandas_code, expected_ref in test_cases:
        code = f"result = {pandas_code}"
        module = astroid.parse(code)
        discovery = ChainDiscovery()
        chains = discovery.discover_chains(module)
        
        assert len(chains) == 1
        assert chains[0].dataframe_reference == expected_ref


def test_registry_filtering():
    """Test registry filtering capabilities."""
    code = """
result1 = df.filter(items=['A'])
result2 = df.rename(columns={'A': 'B'})
result3 = df.groupby('A').mean()
result4 = df[df.A > 1].groupby('B').sum()
"""
    module = astroid.parse(code)
    discovery = ChainDiscovery()
    chains = discovery.discover_chains(module)
    
    registry = discovery.registry
    
    # Test filtering by type
    filter_chains = registry.get_chains_by_type(ChainType.FILTER)
    transform_chains = registry.get_chains_by_type(ChainType.TRANSFORM)
    aggregation_chains = registry.get_chains_by_type(ChainType.AGGREGATION)
    mixed_chains = registry.get_chains_by_type(ChainType.MIXED)
    
    assert len(filter_chains) == 1
    assert len(transform_chains) == 1
    assert len(aggregation_chains) == 1
    assert len(mixed_chains) == 1
    
    # Test filtering by complexity
    simple_chains = registry.get_chains_by_complexity(max_score=2.0)
    complex_chains = registry.get_chains_by_complexity(min_score=3.0)
    
    assert len(simple_chains) == 2  # filter and rename
    assert len(complex_chains) == 2  # groupby chains
    
    # Test filtering by DataFrame
    df_chains = registry.get_chains_by_dataframe("df")
    assert len(df_chains) == 4  # All chains use 'df'


def test_registry_integration():
    """Test that ChainDiscovery properly integrates with ChainRegistry."""
    code = "result = df.filter(items=['A']).rename(columns={'A': 'B'})"
    module = astroid.parse(code)
    
    # Create a custom registry
    custom_registry = ChainRegistry()
    discovery = ChainDiscovery(registry=custom_registry)
    
    chains = discovery.discover_chains(module)
    
    # Check that the chain was registered
    assert len(custom_registry.chains) == 1
    assert custom_registry.chains[0] == chains[0]
    
    # Check that metadata was populated
    chain = chains[0]
    assert chain.chain_type == ChainType.MIXED
    assert chain.complexity_score == 2.0
    assert chain.dataframe_reference == "df"


def test_empty_registry():
    """Test registry behavior with no chains."""
    registry = ChainRegistry()
    
    assert len(registry.chains) == 0
    assert len(registry.get_chains_by_type(ChainType.FILTER)) == 0
    assert len(registry.get_chains_by_complexity()) == 0
    assert len(registry.get_chains_by_dataframe("df")) == 0 