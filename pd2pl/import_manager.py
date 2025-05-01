from typing import List, Tuple
from .import_strategy import ImportStrategy
from .config import TranslationConfig
import re

class ImportManager:
    """
    Manages import handling during pandas to polars translation.
    Implements different strategies for handling imports based on configuration.
    """
    
    def __init__(self, config: TranslationConfig):
        """
        Initialize the ImportManager with translation configuration.
        
        Args:
            config: TranslationConfig instance containing import strategy settings
        """
        self.config = config
        self._required_imports = ["import polars as pl"]
        self._selector_imports = ["import polars.selectors as cs"]
        
    def process_imports(self, code: str) -> str:
        """
        Process imports according to the configured strategy.
        
        Args:
            code: Input code to process
            
        Returns:
            Processed code with imports handled according to strategy
        """
        # Handle empty or whitespace-only code
        if not code.strip():
            return ""
            
        strategy = self.config.import_strategy
        
        if strategy == ImportStrategy.NEVER:
            return code
            
        if strategy == ImportStrategy.AUTO:
            return self._handle_auto_strategy(code)
        elif strategy == ImportStrategy.ALWAYS:
            return self._handle_always_strategy(code)
        elif strategy == ImportStrategy.PRESERVE:
            return self._handle_preserve_strategy(code)
        else:
            raise ValueError(f"Invalid import strategy: {strategy}")
            
    def _handle_auto_strategy(self, code: str) -> str:
        """
        Automatically determine if imports are needed based on code content.
        
        Args:
            code: Input code to analyze
            
        Returns:
            Code with imports added if needed
        """
        if "polars" in code or "pl." in code:
            return code
            
        lines = code.split('\n')
        docstring, imports, rest = self._split_code_components(lines)
        
        # Check if we need selector imports
        needs_selectors = any("select_dtypes" in line for line in rest)
        
        result = []
        if docstring:
            result.extend(docstring)
            result.append('')
            
        if imports:
            result.extend(imports)
            result.append('')
            
        result.extend(self._required_imports)
        if needs_selectors:
            result.extend(self._selector_imports)
            
        result.append('')
        result.extend(rest)
        return '\n'.join(result)
        
    def _handle_always_strategy(self, code: str) -> str:
        """
        Always add required imports regardless of input.
        
        Args:
            code: Input code to process
            
        Returns:
            Code with imports always added
        """
        lines = code.split('\n')
        docstring, imports, rest = self._split_code_components(lines)
        
        result = []
        if docstring:
            result.extend(docstring)
            result.append('')
            
        if imports:
            result.extend(imports)
            result.append('')
            
        result.extend(self._required_imports)
        result.extend(self._selector_imports)  # Always add selectors
        result.append('')
        result.extend(rest)
        return '\n'.join(result)
        
    def _handle_preserve_strategy(self, code: str) -> str:
        """
        Preserve existing import structure while replacing pandas imports.
        
        Args:
            code: Input code to process
            
        Returns:
            Code with pandas imports replaced and other imports preserved
        """
        lines = code.split('\n')
        docstring, imports, rest = self._split_code_components(lines)
        
        # Filter out pandas imports (preserve only non-pandas imports)
        preserved_imports = [line for line in imports if not self._is_pandas_import(line)]
        
        result = []
        if docstring:
            result.extend(docstring)
            result.append('')
        if preserved_imports:
            result.extend(preserved_imports)
            result.append('')
        result.extend(self._required_imports)
        if any("select_dtypes" in line for line in rest):
            result.extend(self._selector_imports)
        result.append('')
        result.extend(rest)
        return '\n'.join(result)
        
    def _split_code_components(self, lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Split code into docstring, imports, and other lines.
        
        Args:
            lines: List of code lines
            
        Returns:
            Tuple of (docstring_lines, import_lines, other_lines)
        """
        docstring = []
        imports = []
        rest = []
        in_docstring = False
        
        for line in lines:
            stripped = line.strip()
            
            # Handle docstrings
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    docstring = []
                in_docstring = not in_docstring
                docstring.append(line)
                continue
                
            if in_docstring:
                docstring.append(line)
                continue
                
            # Handle imports
            if (stripped.startswith(('import ', 'from ')) and 
                not stripped.startswith(('#', '"""', "'''"))):
                imports.append(line)
            else:
                rest.append(line)
                
        return docstring, imports, rest
        
    def _is_pandas_import(self, line: str) -> bool:
        """
        Check if a line is a pandas import.
        
        Args:
            line: Line to check
            
        Returns:
            True if the line is a pandas import, False otherwise
        """
        stripped = line.strip()
        # Regex patterns for common pandas import styles
        patterns = [
            r'^import\s+pandas(\s+as\s+\w+)?',
            r'^import\s+pd',
            r'^from\s+pandas(\.|\s|$)',
            r'^from\s+pd(\.|\s|$)',
        ]
        for pat in patterns:
            if re.match(pat, stripped):
                return True
        return False 