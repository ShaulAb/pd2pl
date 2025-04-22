"""Parser module for converting Python code to AST."""
import ast
from typing import Union

from .errors import ParsingError


def parse_code(code_string: str) -> ast.AST:
    """Parse Python code string into an AST.
    
    Args:
        code_string: String containing Python code to parse
        
    Returns:
        ast.AST: Abstract Syntax Tree representation of the code
        
    Raises:
        ParsingError: If the code contains invalid Python syntax
    """
    try:
        return ast.parse(code_string)
    except SyntaxError as e:
        raise ParsingError(f"Invalid Python syntax: {str(e)}") from e 