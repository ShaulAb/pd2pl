"""pandas_to_polars_translator - A tool to translate pandas code to polars code."""
import ast
import textwrap

from .translator import PandasToPolarsTransformer
from .errors import (
    TranslationError,
    UnsupportedPandasUsageError,
    ParsingError,
)
from .logging import logger
from .imports_postprocess import process_imports
from .config import TranslationConfig

__version__ = "0.1.0"

def translate_code(pandas_code: str, postprocess_imports: bool = False, format_output: bool = False, config: dict = None) -> str:
    """Translate a snippet of pandas code to its polars equivalent.
    
    Args:
        pandas_code: String containing pandas code to translate
        postprocess_imports: If True, add/deduplicate polars imports in the output (default False)
        format_output: If True, format the output code with Black (default False)
        config: Optional configuration dictionary for the translator
        
    Returns:
        str: Translated polars code
        
    Raises:
        ParsingError: If the input code contains invalid Python syntax
        UnsupportedPandasUsageError: If the code uses unsupported pandas features
        TranslationError: If the code cannot be translated for other reasons
    """
    logger.info(f"Original pandas code:\n{textwrap.indent(pandas_code, '  ')}")
    
    try:
        tree = ast.parse(pandas_code)
        logger.debug("AST structure:\n" + ast.dump(tree, indent=2))
    except SyntaxError as e:
        logger.error(f"Failed to parse pandas code: {e}")
        raise ParsingError("Invalid pandas code provided") from e

    effective_config = config or TranslationConfig.get_config()
    transformer = PandasToPolarsTransformer(config=effective_config)
    try:
        logger.debug(f">>> translate_code: About to visit AST tree: {repr(tree)}")
        new_tree = transformer.visit(tree)
    except Exception as e:
        logger.error(f"Error during AST transformation: {e}", exc_info=True)
        raise
    
    # Add required imports
    imports = []
    if getattr(transformer, 'needs_polars_import', False):
        imports.append("import polars as pl")
    if getattr(transformer, 'needs_selector_import', False):
        imports.append("import polars.selectors as cs")
    
    polars_code = ast.unparse(new_tree)
    
    if postprocess_imports:
        polars_code = process_imports(polars_code)
    if format_output:
        try:
            import black
            polars_code = black.format_str(polars_code, mode=black.FileMode())
        except ImportError:
            logger.warning("Black is not installed; skipping code formatting.")
        except Exception as e:
            logger.warning(f"Black formatting failed: {e}")
    return polars_code
    # Note: The import calculation logic remains but is currently unused in the return value
