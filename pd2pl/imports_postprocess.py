import re
from typing import List
from enum import Enum, auto

class ImportStrategy(Enum):
    ALWAYS = auto()
    NEVER = auto()
    AUTO = auto()
    PRESERVE = auto()

def _has_import(code_lines: List[str], import_stmt: str) -> bool:
    return any(line.strip() == import_stmt for line in code_lines)

def _find_import_insertion_index(code_lines: List[str]) -> int:
    """
    Returns the index after the module docstring (if present),
    or after initial comments, or 0 if neither.
    """
    idx = 0
    n = len(code_lines)
    # Skip initial comments
    while idx < n and code_lines[idx].strip().startswith('#'):
        idx += 1
    # Check for module docstring
    if idx < n and re.match(r'\s*[\'\"]{3}', code_lines[idx]):
        docstring_line = code_lines[idx].strip()
        # Single-line docstring
        if docstring_line.count('"""') == 2 or docstring_line.count("'''") == 2:
            idx += 1
        else:
            # Multi-line docstring
            idx += 1
            while idx < n and not re.match(r'.*[\'\"]{3}', code_lines[idx]):
                idx += 1
            if idx < n:
                idx += 1  # Skip closing docstring line
    return idx

def _deduplicate_imports(code_lines: List[str]) -> List[str]:
    seen = set()
    result = []
    for line in code_lines:
        if line.strip().startswith('import') or line.strip().startswith('from'):
            if line.strip() not in seen:
                seen.add(line.strip())
                result.append(line)
        else:
            result.append(line)
    return result

def _remove_pandas_import(code_lines: List[str]) -> List[str]:
    return [line for line in code_lines if not re.match(r'\s*import\s+pandas\s+as\s+pd', line)]

def _format_output(code: str) -> str:
    try:
        import black
        return black.format_str(code, mode=black.Mode())
    except ImportError:
        # Black not installed; return code unchanged
        return code
    except Exception:
        # Any formatting error, return code unchanged
        return code

def process_imports(
    code: str,
    import_strategy: ImportStrategy = ImportStrategy.AUTO,
    needs_polars_import: bool = False,
    needs_selector_import: bool = False,
    format_output: bool = False
) -> str:
    """
    Post-processes code to manage import statements according to the selected strategy.
    """
    STRATEGY_HANDLERS = {
        ImportStrategy.ALWAYS: _process_always_strategy,
        ImportStrategy.NEVER: _process_never_strategy,
        ImportStrategy.AUTO: _process_auto_strategy,
        ImportStrategy.PRESERVE: _process_preserve_strategy,
    }
    handler = STRATEGY_HANDLERS[import_strategy]
    return handler(
        code,
        needs_polars_import=needs_polars_import,
        needs_selector_import=needs_selector_import,
        format_output=format_output
    )

def _process_always_strategy(code, needs_polars_import=False, needs_selector_import=False, format_output=False):
    """
    Always add required polars imports, deduplicate, remove pandas, format if requested.
    """
    code_lines = code.splitlines()
    imports_to_add = []
    # Always add polars imports
    if not _has_import(code_lines, 'import polars as pl'):
        imports_to_add.append('import polars as pl')
    if needs_selector_import and not _has_import(code_lines, 'import polars.selectors as cs'):
        imports_to_add.append('import polars.selectors as cs')
    if imports_to_add:
        insert_idx = _find_import_insertion_index(code_lines)
        code_lines = code_lines[:insert_idx] + imports_to_add + code_lines[insert_idx:]
    # Remove pandas import
    code_lines = _remove_pandas_import(code_lines)
    # Deduplicate imports
    code_lines = _deduplicate_imports(code_lines)
    # Ensure a single blank line after the last import (if any imports present)
    last_import_idx = -1
    for idx, line in enumerate(code_lines):
        if line.strip().startswith('import') or line.strip().startswith('from'):
            last_import_idx = idx
    if last_import_idx != -1 and last_import_idx + 1 < len(code_lines):
        if code_lines[last_import_idx + 1].strip() != '':
            code_lines.insert(last_import_idx + 1, '')
    result = '\n'.join(code_lines)
    if format_output:
        result = _format_output(result)
    # Ensure import datetime if datetime.datetime is used
    if 'datetime.datetime' in result and 'import datetime' not in result:
        result = 'import datetime\n' + result
    return result

def _process_never_strategy(code, **kwargs):
    """
    Never add or modify imports, return code unchanged.
    """
    # Never add or modify imports
    return code

def _process_auto_strategy(code, needs_polars_import=False, needs_selector_import=False, format_output=False):
    """
    Add polars imports only if needed, deduplicate, remove pandas, format if requested.
    """
    code_lines = code.splitlines()
    imports_to_add = []
    if needs_polars_import and not _has_import(code_lines, 'import polars as pl'):
        imports_to_add.append('import polars as pl')
    if needs_selector_import and not _has_import(code_lines, 'import polars.selectors as cs'):
        imports_to_add.append('import polars.selectors as cs')
    if imports_to_add:
        insert_idx = _find_import_insertion_index(code_lines)
        code_lines = code_lines[:insert_idx] + imports_to_add + code_lines[insert_idx:]
    code_lines = _remove_pandas_import(code_lines)
    code_lines = _deduplicate_imports(code_lines)
    last_import_idx = -1
    for idx, line in enumerate(code_lines):
        if line.strip().startswith('import') or line.strip().startswith('from'):
            last_import_idx = idx
    if last_import_idx != -1 and last_import_idx + 1 < len(code_lines):
        if code_lines[last_import_idx + 1].strip() != '':
            code_lines.insert(last_import_idx + 1, '')
    result = '\n'.join(code_lines)
    if format_output:
        result = _format_output(result)
    if 'datetime.datetime' in result and 'import datetime' not in result:
        result = 'import datetime\n' + result
    return result

def _process_preserve_strategy(code, needs_polars_import=False, needs_selector_import=False, format_output=False):
    """
    Preserve all existing imports, including pandas, even if unused. Only add polars imports if needed, but never remove or replace pandas import.
    """
    code_lines = code.splitlines()
    new_code_lines = []
    pandas_import_pattern = re.compile(r'\s*import\s+pandas\s+as\s+pd')
    polars_added = False
    selectors_added = False
    for line in code_lines:
        # Always preserve pandas import
        if pandas_import_pattern.match(line):
            new_code_lines.append(line)
            continue
        new_code_lines.append(line)
    # Add polars imports if needed, but do not remove pandas
    insert_idx = _find_import_insertion_index(new_code_lines)
    imports_to_add = []
    if needs_polars_import and not _has_import(new_code_lines, 'import polars as pl'):
        imports_to_add.append('import polars as pl')
    if needs_selector_import and not _has_import(new_code_lines, 'import polars.selectors as cs'):
        imports_to_add.append('import polars.selectors as cs')
    if imports_to_add:
        new_code_lines = new_code_lines[:insert_idx] + imports_to_add + new_code_lines[insert_idx:]
    result = '\n'.join(new_code_lines)
    if format_output:
        result = _format_output(result)
    return result 