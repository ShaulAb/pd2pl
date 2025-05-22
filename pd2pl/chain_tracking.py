"""Chain tracking system for method transformations.

This module identifies and tracks method chains to enable proper
schema propagation and method transformations across complex chains.
"""
from typing import Dict, List, Set, Optional, Any, Tuple
import ast
from collections import defaultdict
import uuid

from pd2pl.logging import logger
from pd2pl.schema_tracking import SchemaState, SchemaRegistry


class ChainNode:
    """Represents a node in a method chain."""
    def __init__(self, node_id: int, method_name: str, node: ast.Call):
        self.node_id = node_id
        self.method_name = method_name
        self.node = node
        self.parent: Optional['ChainNode'] = None
        self.children: List['ChainNode'] = []
        self.chain_id: Optional[str] = None
        self.position: int = 0
        self.schema_before: Optional[SchemaState] = None
        self.schema_after: Optional[SchemaState] = None
        
    def add_child(self, child: 'ChainNode') -> None:
        """Add a child node to this chain node."""
        child.parent = self
        self.children.append(child)
        
    def set_chain_id(self, chain_id: str, position: int = 0) -> None:
        """Set chain ID and position for this node."""
        self.chain_id = chain_id
        self.position = position
        # Propagate to children
        for i, child in enumerate(self.children):
            child.set_chain_id(chain_id, position + i + 1)
            
    def __repr__(self) -> str:
        return f"ChainNode({self.method_name}, id={self.node_id}, chain={self.chain_id}, pos={self.position})"


class ChainRegistry:
    """Registry for tracking method chains throughout AST transformation."""
    def __init__(self):
        self.nodes_by_id: Dict[int, ChainNode] = {}
        self.chains_by_id: Dict[str, List[ChainNode]] = defaultdict(list)
        self.root_nodes: List[ChainNode] = []
        
    def register_node(self, node: ast.Call, method_name: str) -> ChainNode:
        """Register a node in the chain registry."""
        node_id = id(node)
        chain_node = ChainNode(node_id, method_name, node)
        self.nodes_by_id[node_id] = chain_node
        return chain_node
        
    def add_to_chain(self, parent_node: Optional[ast.AST], child_node: ChainNode) -> None:
        """Add a child node to a parent in the chain."""
        if parent_node is not None:
            parent_id = id(parent_node)
            if parent_id in self.nodes_by_id:
                parent = self.nodes_by_id[parent_id]
                parent.add_child(child_node)
            else:
                # Parent not registered yet, add as root for now
                self.root_nodes.append(child_node)
        else:
            # This is a root node
            self.root_nodes.append(child_node)
            
    def get_node(self, node: ast.AST) -> Optional[ChainNode]:
        """Get chain information for a node."""
        return self.nodes_by_id.get(id(node))
        
    def finalize_chains(self) -> None:
        """Finalize all chains by assigning chain IDs and positions."""
        for root in self.root_nodes:
            chain_id = str(uuid.uuid4())[:8]  # Short ID for readability in logs
            root.set_chain_id(chain_id)
            # Collect all nodes in this chain
            self._collect_chain_nodes(root, chain_id)
            
    def _collect_chain_nodes(self, node: ChainNode, chain_id: str) -> None:
        """Collect all nodes in a chain."""
        self.chains_by_id[chain_id].append(node)
        for child in node.children:
            self._collect_chain_nodes(child, chain_id)
            
    def is_in_chain(self, node: ast.AST) -> bool:
        """Check if a node is part of a method chain."""
        return id(node) in self.nodes_by_id
        
    def get_chain_for_node(self, node: ast.AST) -> Optional[List[ChainNode]]:
        """Get the complete chain for a node."""
        chain_node = self.get_node(node)
        if not chain_node or not chain_node.chain_id:
            return None
        return self.chains_by_id.get(chain_node.chain_id)
    
    def get_root_for_chain(self, chain_id: str) -> Optional[ChainNode]:
        """Get the root node for a chain."""
        nodes = self.chains_by_id.get(chain_id, [])
        if not nodes:
            return None
        return min(nodes, key=lambda n: n.position)
        
    def print_chains(self) -> None:
        """Print all chains for debugging."""
        logger.debug(f"Chain Registry contains {len(self.chains_by_id)} chains:")
        for chain_id, nodes in self.chains_by_id.items():
            sorted_nodes = sorted(nodes, key=lambda n: n.position)
            chain_str = " -> ".join(f"{n.method_name}[{n.position}]" for n in sorted_nodes)
            logger.debug(f"Chain {chain_id}: {chain_str}")
            
            # Print schema information if available
            for node in sorted_nodes:
                if node.schema_before:
                    cols_before = node.schema_before.columns if node.schema_before else set()
                    logger.debug(f"  {node.method_name}[{node.position}] schema before: cols={cols_before}")
                if node.schema_after:
                    cols_after = node.schema_after.columns if node.schema_after else set()
                    logger.debug(f"  {node.method_name}[{node.position}] schema after: cols={cols_after}") 