import ast

from pd2pl.translator import PandasToPolarsTransformer
from loguru import logger

def translate_with_debug(code):
    """Helper to translate pandas code to polars with debug output."""
    logger.debug(f"Input code: {code}")
    tree = ast.parse(code)
    
    # Debug the AST before transformation
    logger.debug(f"AST before transformation: {ast.dump(tree)}")
    
    # Create transformer with schema tracking
    transformer = PandasToPolarsTransformer()
    
    # Transform the code with preprocessing
    transformed = transformer.process(tree)
    result = ast.unparse(transformed)
    
    # Debug the AST after transformation
    logger.debug(f"AST after transformation: {ast.dump(transformed)}")
    
    logger.debug(f"Output code: {result}")
    
    # Debug output of schemas
    logger.debug("Final schema registry state:")
    for var_name, schema in transformer.schema_registry.schemas.items():
        logger.debug(f"  Variable: {var_name}")
        logger.debug(f"    Columns: {schema.columns}")
        logger.debug(f"    Group keys: {schema.group_keys}")
        logger.debug(f"    In groupby chain: {schema.in_groupby_chain}")
        logger.debug(f"    Aggregated columns: {schema.aggregated_columns}")
    
    # Debug output of chains
    if transformer.chain_registry:
        logger.debug("Chain structure:")
        transformer.chain_registry.print_chains()
    
    return result

def test_basic_groupby_sort_chain():
    """Test a basic groupby + mean + sort_values chain."""
    # Simple test case with debug output
    pandas_code = """df[["A", "B"]].groupby("A").mean().sort_values(by="B", ascending=False)"""
    
    polars_code = translate_with_debug(pandas_code)
    
    # Print the result for inspection
    print(f"\nTranslation result:\n{polars_code}\n")
    
    # Check that we correctly translated to B_mean in the sort call
    assert "B_mean" in polars_code
    assert "descending=True" in polars_code
    # Check method names were correctly translated
    assert ".group_by" in polars_code
    assert ".sort" in polars_code 