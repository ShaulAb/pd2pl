"""pandas_to_polars_translator - A tool to translate pandas code to polars code."""
import ast
import textwrap
from typing import Optional, Union

from .translator import PandasToPolarsTransformer, translate_code
from .errors import (
    TranslationError,
    UnsupportedPandasUsageError,
    ParsingError,
)
from .logging import logger
from .imports_postprocess import process_imports
from .config import TranslationConfig
from .import_strategy import ImportStrategy

__version__ = "0.1.0"

__all__ = ['translate_code', 'TranslationConfig', 'ImportStrategy']

# Re-export the translate_code function with updated docstring
def translate_code(
    code: str,
    config: Optional[TranslationConfig] = None,
    import_strategy: Union[ImportStrategy, str] = ImportStrategy.AUTO,
    postprocess_imports: bool = True,  # For backward compatibility
    format_output: bool = True
) -> str:
    """Translate a snippet of pandas code to its polars equivalent.
    
    Args:
        code: String containing pandas code to translate
        config: Optional TranslationConfig instance for fine-grained control
        import_strategy: Strategy for handling imports during translation:
            - AUTO: Automatically determine if imports are needed
            - ALWAYS: Always add required imports
            - NEVER: Never add imports
            - PRESERVE: Preserve existing import structure
        postprocess_imports: Whether to postprocess imports (deprecated, use import_strategy instead)
        format_output: Whether to format the output code with Black
        
    Returns:
        str: Translated polars code
        
    Raises:
        ParsingError: If the input code contains invalid Python syntax
        UnsupportedPandasUsageError: If the code uses unsupported pandas features
        TranslationError: If the code cannot be translated for other reasons
    """
    from .translator import translate_code as _translate_code
    return _translate_code(
        code=code,
        config=config,
        import_strategy=import_strategy,
        postprocess_imports=postprocess_imports,
        format_output=format_output
    )
