"""New transformation engine for converting pandas chains to polars using astroid-based discovery.

This module provides a clean, test-driven transformation engine that:
1. Consumes ChainRegistry and AstroidSchemaRegistry from our discovery phase
2. Transforms pandas method chains to polars equivalents
3. Maintains schema awareness throughout transformation
4. Follows clean architecture principles without legacy debt
"""
import ast
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .chain_discovery import ChainRegistry, MethodChain
from .schema_tracking import AstroidSchemaRegistry, AstroidSchemaState


@dataclass
class ChainTransformation:
    """Metadata about a chain transformation."""
    original_chain: MethodChain
    transformation_type: str
    methods_transformed: List[str]


@dataclass
class TransformationResult:
    """Result of transforming a pandas chain to polars using pure astroid."""
    transformed_code: str  # Pure astroid: return source code, not AST
    polars_imports_needed: List[str]
    warnings: List[str]
    schema_changes: Dict[str, Any]
    chain_transformations: List[ChainTransformation] = field(default_factory=list)


class ChainTransformer:
    """Clean transformation engine for pandas→polars chain conversion."""
    
    def __init__(self, chain_registry: ChainRegistry, schema_registry: AstroidSchemaRegistry):
        """Initialize transformer with discovery results.
        
        Args:
            chain_registry: Registry of discovered method chains
            schema_registry: Registry of schema states and evolution
        """
        self.chain_registry = chain_registry
        self.schema_registry = schema_registry
        self.polars_imports_needed = set()
        self.warnings = []
        
    def transform_chains(self, source_code: str) -> TransformationResult:
        """Transform all discovered chains using pure astroid approach.
        
        Args:
            source_code: The source code containing pandas operations
            
        Returns:
            TransformationResult with transformed code and metadata
        """
        # Parse with astroid for transformation
        import astroid
        astroid_tree = astroid.parse(source_code)
        
        # Apply basic pandas→polars transformations using astroid
        transformer = PandasToPolarsAstroidTransformer()
        transformed_code = transformer.transform(astroid_tree)
        
        # Track imports needed
        if transformer.needs_polars_import:
            self.polars_imports_needed.add("polars")
        
        # Process discovered chains and create transformation metadata
        chain_transformations = self._transform_discovered_chains()
        
        # Pure astroid: return transformed source code directly
        return TransformationResult(
            transformed_code=transformed_code,
            polars_imports_needed=list(self.polars_imports_needed),
            warnings=self.warnings.copy(),
            schema_changes=self._collect_schema_changes(),
            chain_transformations=chain_transformations
        )
    
    def _transform_single_chain(self, ast_tree: ast.AST, chain: MethodChain) -> ast.AST:
        """Transform a single method chain to polars equivalent.
        
        Args:
            ast_tree: Current AST tree
            chain: Method chain to transform
            
        Returns:
            Updated AST tree with chain transformed
        """
        # For now, return the tree unchanged - we'll implement step by step
        return ast_tree
    
    def _collect_schema_changes(self) -> Dict[str, Any]:
        """Collect schema changes from all transformations."""
        return {}
    
    def _transform_discovered_chains(self) -> List[ChainTransformation]:
        """Transform discovered chains and return transformation metadata."""
        from .chain_discovery import ChainType
        
        chain_transformations = []
        
        for chain in self.chain_registry.get_all_chains():
            # Determine transformation type based on chain classification
            transformation_type = self._get_transformation_type(chain)
            
            # Create transformation metadata for each discovered chain
            transformation = ChainTransformation(
                original_chain=chain,
                transformation_type=transformation_type,
                methods_transformed=[method.name for method in chain.methods]
            )
            chain_transformations.append(transformation)
        
        return chain_transformations
    
    def _get_transformation_type(self, chain: MethodChain) -> str:
        """Determine the appropriate transformation type for a chain."""
        from .chain_discovery import ChainType
        
        if chain.chain_type == ChainType.AGGREGATION:
            return "aggregation"
        elif chain.chain_type == ChainType.FILTER:
            return "filter"
        elif chain.chain_type == ChainType.TRANSFORM:
            return "transform"
        elif chain.chain_type == ChainType.MIXED:
            return "mixed"
        else:
            return "basic_preservation"


class PandasToPolarsAstroidTransformer:
    """Pure astroid transformer for basic pandas→polars conversions using proper astroid visitors."""
    
    def __init__(self):
        self.needs_polars_import = False
        self.transformations = []
    
    def transform(self, astroid_tree) -> str:
        """Transform astroid tree using proper astroid node visitors."""
        from astroid import nodes
        
        # Visit all nodes to find transformation opportunities
        for node in astroid_tree.nodes_of_class((nodes.Import, nodes.ImportFrom, nodes.Attribute, nodes.Call)):
            self._visit_node(node)
        
        # Apply transformations to source code
        source_code = astroid_tree.as_string()
        
        # Apply transformations in reverse order to maintain line positions
        for original, replacement in reversed(self.transformations):
            source_code = source_code.replace(original, replacement)
        
        return source_code
    
    def _visit_node(self, node):
        """Visit individual nodes and record transformations."""
        from astroid import nodes
        
        if isinstance(node, nodes.Import):
            self._visit_import(node)
        elif isinstance(node, nodes.ImportFrom):
            self._visit_import_from(node)
        elif isinstance(node, nodes.Attribute):
            self._visit_attribute(node)
        elif isinstance(node, nodes.Call):
            self._visit_call(node)
    
    def _visit_import(self, node):
        """Handle 'import pandas as pd' style imports."""
        from astroid import nodes
        
        for alias in node.names:
            if alias[0] == 'pandas':  # alias is a tuple (name, asname)
                if alias[1] == 'pd':  # import pandas as pd
                    self.transformations.append(('import pandas as pd', 'import polars as pl'))
                    self.needs_polars_import = True
                elif alias[1] is None:  # import pandas
                    self.transformations.append(('import pandas', 'import polars'))
                    self.needs_polars_import = True
    
    def _visit_import_from(self, node):
        """Handle 'from pandas import ...' style imports."""
        if node.modname == 'pandas':
            self.transformations.append((f'from pandas', 'from polars'))
            self.needs_polars_import = True
    
    def _visit_attribute(self, node):
        """Handle attribute access like pd.DataFrame."""
        from astroid import nodes
        
        # Check for pd.DataFrame pattern
        if (isinstance(node.expr, nodes.Name) and 
            node.expr.name == 'pd' and 
            node.attrname == 'DataFrame'):
            self.transformations.append(('pd.DataFrame', 'pl.DataFrame'))
            self.needs_polars_import = True
    
    def _visit_call(self, node):
        """Handle method calls like df.groupby().mean() using flexible pattern matching."""
        from astroid import nodes
        import re
        
        # Convert the call node back to source code to analyze the pattern
        call_source = node.as_string()
        
        # Handle aggregation pattern: {dataframe}.groupby({column}).{agg_method}()
        groupby_agg_pattern = r'(\w+)\.groupby\(([^)]+)\)\.(\w+)\(\)'
        match = re.match(groupby_agg_pattern, call_source)
        
        if match:
            dataframe_name = match.group(1)
            column_spec = match.group(2)  # Could be 'A' or ['A', 'B'] etc.
            agg_method = match.group(3)
            
            # Transform to polars syntax
            polars_equivalent = f"{dataframe_name}.group_by({column_spec}).agg(pl.all().{agg_method}())"
            self.transformations.append((call_source, polars_equivalent))
            self.needs_polars_import = True


def create_transformer(source_code: str) -> ChainTransformer:
    """Factory function to create a transformer with discovery phase.
    
    Args:
        source_code: Python source code containing pandas operations
        
    Returns:
        ChainTransformer ready for transformation
    """
    # Import here to avoid circular imports
    from .chain_discovery import ChainDiscovery
    from .schema_tracking import AstroidSchemaRegistry
    import astroid
    
    # Parse directly to astroid - no bridge needed
    astroid_tree = astroid.parse(source_code)
    
    # Run discovery phase
    schema_registry = AstroidSchemaRegistry()
    discovery = ChainDiscovery(schema_registry=schema_registry)
    chains = discovery.discover_chains(astroid_tree)
    
    # Get the registry with registered chains
    chain_registry = discovery.registry
    
    return ChainTransformer(chain_registry, schema_registry) 